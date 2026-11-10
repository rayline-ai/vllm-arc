"""The ARC release's Stage-1 head as torch modules.

The ARC training SDK (not yet public) compiles a fitted Stage-1 selector into tensors and checks this module against
the selector's own ``predict``; this plugin loads the same tensors from a release's safetensors (``vllm_arc.heads``)
and runs this module in the vLLM process. It imports torch only. The constants below restate the SDK's, which the
SDK's tests pin equal.
"""

from __future__ import annotations

import torch

#: The usage orderer's thinnest usable pool and its token floor.
MIN_POOL = 20
FLOOR_TOKENS = 1.0
#: A Platt map's clip of p away from 0 and 1.
PLATT_EPS = 1e-6
#: Where the prompt's ``log1p`` token count sits in
#: a request context-feature vector (the window term's numerator).
PROMPT_FEATURE = 0


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """`numpy.interp` of ``x[..., k]`` on each ``xp[..., :n]`` / ``fp[..., :n]`` (constant beyond the end knots)."""
    last = (n - 1).clamp(min=0)[..., None]
    count = n[..., None]
    right = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True)
    j = torch.minimum((right - 1).clamp(min=0), (last - 1).clamp(min=0))
    j1 = torch.minimum(j + 1, last)
    x0, x1 = torch.gather(xp, -1, j), torch.gather(xp, -1, j1)
    f0, f1 = torch.gather(fp, -1, j), torch.gather(fp, -1, j1)
    span = torch.where(x1 > x0, x1 - x0, torch.ones_like(x1))
    inner = (f1 - f0) / span * (x - x0) + f0
    first, final = fp[..., :1].expand_as(x), torch.gather(fp, -1, last).expand_as(x)
    out = torch.where(x < xp[..., :1], first, torch.where(x >= torch.gather(xp, -1, last), final, inner))
    return torch.where(count > 1, out, first)


def _linear_quantile(pool: torch.Tensor, n: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """NumPy's ``quantile(method="linear")`` of each ``pool[..., :n]`` at ``q[..., k]``, bit for bit: virtual index
    ``(n - 1) q``, its floor and the next index clipped into the pool, then NumPy's two-sided ``_lerp``."""
    last = (n - 1).clamp(min=0)[..., None]
    virtual = (n - 1)[..., None].to(q.dtype) * q
    below = torch.floor(virtual)
    previous = torch.minimum(below.clamp(min=0).to(torch.int64), last)
    following = torch.minimum(previous + 1, last)
    a, b = torch.gather(pool, -1, previous), torch.gather(pool, -1, following)
    gamma = virtual - below
    diff = b - a
    return torch.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)


def order_ratio(glm: torch.Tensor, net: torch.Tensor, pool_glm: torch.Tensor, pool_net: torch.Tensor,
                pool_n: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """The usage orderer's ratio over ``glm`` / ``net`` ``[C, L, rows]`` against the padded ``[C, L, P]`` pools:
    1 where either value is non-finite or the pool is shorter than `MIN_POOL`."""
    finite = torch.isfinite(glm) & torch.isfinite(net)
    count = pool_n.clamp(min=1)[..., None].to(glm.dtype)
    u = torch.searchsorted(pool_net, torch.where(finite, net, 0.0).contiguous(), right=True).to(glm.dtype) / count
    v = torch.searchsorted(pool_glm, torch.where(finite, glm, 0.0).contiguous(), right=True).to(glm.dtype) / count
    top = torch.clamp(_linear_quantile(pool_glm, pool_n, u.clamp(0, 1)), min=FLOOR_TOKENS)
    bottom = torch.clamp(_linear_quantile(pool_glm, pool_n, v.clamp(0, 1)), min=FLOOR_TOKENS)
    thick = (pool_n >= MIN_POOL)[..., None]
    return torch.where(finite & thick, torch.pow(top / bottom, alpha), torch.ones_like(glm))


def _platt(p: torch.Tensor, on: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A cross-fit calibration map, for Platt maps: ``sigmoid(a logit(clip(p)) + b)`` where ``on``
    and p is finite, else p unchanged."""
    q = p.clamp(PLATT_EPS, 1 - PLATT_EPS)
    z = a * torch.log(q / (1 - q)) + b
    mapped = torch.exp(-torch.logaddexp(torch.zeros_like(z), -z))
    return torch.where(on.bool() & torch.isfinite(p), mapped, p)


def apply_shrink(p: torch.Tensor, ref: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """``p[M, n, C]`` with each candidate c whose ``ref[c]`` is non-empty pulled toward the mean of its reference
    candidates' finite p in that row; a row with no finite reference leaves c as predicted, and NaN stays NaN."""
    finite = ~torch.isnan(p)
    total = torch.einsum("mnj,cj->mnc", torch.where(finite, p, torch.zeros_like(p)), ref)
    count = torch.einsum("mnj,cj->mnc", finite.to(p.dtype), ref)
    pulled = (1 - strength) * p + strength * total / count.clamp(min=1.0)
    return torch.where((ref.sum(1) > 0)[None, None, :] & (count > 0), pulled, p)


def stage_one_module(tensors: dict[str, torch.Tensor]) -> torch.nn.Module:
    """``forward(x[n, d] float64, regime[n] int64[, context[n, q] float64][, cache[n, models, 6] float64])
    -> (p[M, n, C], usage[M, n, C, 4])``, member 0 the point. ``context`` (each row's raw
    request context features, the prompt's ``log1p`` token count first) is required by, and refused without, a context-feature usage head
    (``ctx_mean`` among the tensors). ``cache`` (each row's six warm-cache features per selector model,
    NaN where a value is unknown) is required by, and refused without, a warm-cache floor head (``w_cache``)."""
    class StageOne(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, value in tensors.items():
                self.register_buffer(name, value)

        def forward(self, x: torch.Tensor, regime: torch.Tensor, context: torch.Tensor | None = None,
                    cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            if hasattr(self, "transform_weight"):
                x = x @ self.transform_weight.T + self.transform_bias
            centered = x - self.mean
            z = centered @ self.components.T if hasattr(self, "components") else centered * self.scale
            eta = (self.logit_bias[:, None, :] + self.regime[:, regime][:, :, None]
                   - (z @ self.difficulty.T).T[:, :, None]
                   + torch.einsum("nw,mrw,mcr->mnc", z, self.v, self.u))
            if hasattr(self, "w1"):  # mlp_embed: <relu(z w1 + c1) w2 + c2, embed[c]>
                hidden = torch.relu(torch.einsum("nd,mdh->mnh", z, self.w1) + self.c1[:, None])
                h = torch.einsum("mnh,mhe->mne", hidden, self.w2) + self.c2[:, None]
                eta = eta + torch.einsum("mne,mce->mnc", h, self.embed)
            nan = torch.tensor(float("nan"), dtype=eta.dtype)
            p = torch.where(self.p_ok.bool()[:, None, :], torch.sigmoid(eta.clamp(-40.0, 40.0)), nan)
            if hasattr(self, "shrink_ref"):  # unmeasured candidates toward their tier's finite measured mean
                p = apply_shrink(p, self.shrink_ref, self.shrink_strength)
            exponent = torch.einsum("nw,mckw->mnck", z, self.w_usage) + self.b_usage[:, None]
            s = prompt = None  # the standardized request features and prompt tokens of a context-feature head
            if hasattr(self, "ctx_mean"):
                if context is None:
                    raise ValueError("this head's usage reads request context features: pass context")
                if context.shape != (x.shape[0], self.ctx_mean.shape[0]) or not bool(torch.isfinite(context).all()):
                    raise ValueError("request context features are one finite row per input row, of the head's width")
                s = (context - self.ctx_mean) / self.ctx_scale
                prompt = torch.expm1(context[:, PROMPT_FEATURE])
                exponent = exponent + torch.einsum("nq,mckq->mnck", s, self.w_ctx) \
                    + (prompt[:, None] * self.inv_window[None, :])[None, :, :, None] * self.w_frac[:, None]
            elif context is not None:
                raise ValueError("this head reads no request context features")
            cache_floor = None
            if hasattr(self, "w_cache"):  # warm-cache usage features and their floor
                if cache is None:
                    raise ValueError("this head's usage reads warm-cache features: pass cache")
                if cache.dim() != 3 or cache.shape[0] != x.shape[0] or cache.shape[2] != 6 \
                        or not bool(torch.isfinite(cache[..., :2]).all()):
                    raise ValueError("warm-cache features are rows x models x 6 with known/absent indicators set")
                raw = cache[:, self.get_buffer("cache_model"), :]  # [n, C, 6]: each candidate's model's state
                known = torch.nan_to_num(raw[..., 0])
                if not bool(torch.isfinite(raw[..., [2, 3, 5]][known == 1]).all()):  # a known state is complete
                    raise ValueError("a known warm-cache state carries its prompt tokens, cache-read share and epoch "
                                     "match; partial evidence is unknown, never a measured zero")
                columns = torch.cat([raw[..., :2], torch.nan_to_num(raw[..., [2, 3, 5]]) * known[..., None]], dim=-1)
                exponent = exponent + torch.einsum("ncj,mckj->mnck", columns, self.w_cache)
                ok = (known == 1) & (torch.nan_to_num(raw[..., 5]) == 1) & torch.isfinite(raw[..., 2])
                cache_floor = torch.where(ok, torch.expm1(torch.nan_to_num(raw[..., 2])), torch.zeros_like(known))
            elif cache is not None:
                raise ValueError("this head reads no warm-cache features")
            if hasattr(self, "floor_ratio"):  # the usage floor head
                if s is None or prompt is None:  # a floor head is compiled only with context features
                    raise ValueError("this floor head carries no request context features")
                usage = self._floor(exponent, prompt, nan, s, cache_floor)
            else:
                mean = torch.exp(exponent) * self.smear[:, None] - 1.0
                usage = torch.minimum(torch.maximum(mean, self.low[:, None]), self.high[:, None])
                usage = torch.where(self.usage_ok.bool()[:, None], usage, nan)
            if hasattr(self, "ord_fit_member"):  # each member's own orderer on its own fits (contiguous, in order)
                usage = torch.cat([self._order(z, usage[self.ord_fit_member == k], k)
                                   for k in range(self.ord_mean.shape[0])])
            elif hasattr(self, "ord_mean"):  # the within-model usage orderer
                usage = self._order(z, usage)
            if hasattr(self, "ens_point"):  # a calibrated seed ensemble
                return self._ensemble(p, usage)
            return p, usage

        def _floor(self, exponent: torch.Tensor, prompt: torch.Tensor, nan: torch.Tensor,
                   s: torch.Tensor, cache_floor: torch.Tensor | None = None) -> torch.Tensor:
            """Each leg relative to the prompt (log1p offset), the output under ``floor_calls`` relative to the
            clipped input side instead; then an input side below ``floor_ratio x prompt`` (or, with ``cache_floor``
            [n, C], a known same-epoch last call's prompt where larger) scaled up to it."""
            log_prompt = torch.log1p(prompt)[None, :, None, None]
            head = self.floor_head_ok.bool()[:, None, :, None]  # [M, 1, C, 1]
            head_log = (self.floor_b[:, None, :, 0] + torch.einsum("nq,mcq->mnc", s, self.floor_b[..., 1:]))[..., None]
            log_prompt = torch.where(head, head_log, log_prompt)
            ok = self.usage_ok.bool()[:, None]
            low, high = self.low[:, None], self.high[:, None]
            inputs = torch.exp((exponent[..., :3] + log_prompt).clamp(-40.0, 40.0)) - 1.0
            inputs = torch.where(ok[..., :3], torch.minimum(torch.maximum(inputs, low[..., :3]), high[..., :3]), nan)
            side = torch.nan_to_num(inputs, nan=0.0).sum(-1, keepdim=True)
            calls = bool(self.floor_calls) and bool(ok[..., :3].any())
            out_offset = torch.log(side + 1.0) if calls else log_prompt.expand_as(side)
            output = torch.exp((exponent[..., 3:] + out_offset).clamp(-40.0, 40.0)) - 1.0
            output = torch.where(ok[..., 3:], torch.minimum(torch.maximum(output, low[..., 3:]), high[..., 3:]), nan)
            floor = torch.where(head, torch.expm1(head_log + self.floor_rq[:, None, :, None]),
                                self.floor_ratio[:, None, :, None] * prompt[None, :, None, None])
            if cache_floor is not None:
                floor = torch.maximum(floor, cache_floor[None, :, :, None])
            full = torch.isfinite(inputs).all(-1, keepdim=True)
            scale = torch.where(full & (side < floor) & (side > 0), floor / side.clamp_min(1e-9),
                                torch.ones_like(side))
            inputs = inputs * scale
            empty = full & (side <= 0) & (floor > 0)  # no split to keep: the floor is uncached input
            inputs = torch.cat([torch.where(empty, floor, inputs[..., :1]), inputs[..., 1:]], dim=-1)
            return torch.cat([inputs, output], dim=-1)

        def _ensemble(self, p: torch.Tensor, usage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            points = _platt(p[self.ens_point], self.cal_platt[:, None], self.cal_a[:, None], self.cal_b[:, None])
            finite = torch.isfinite(points)
            count = finite.sum(0)
            total = torch.where(finite, points, torch.zeros_like(points)).sum(0)
            mean = torch.where(count > 0, total / count.clamp(min=1), torch.nan)
            draws = _platt(p[self.ens_draws], self.cal_platt[self.ens_owner][:, None],
                           self.cal_a[self.ens_owner][:, None], self.cal_b[self.ens_owner][:, None])
            usage_mean = usage[self.ens_point].mean(0)  # NaN where any member's is
            return torch.cat([mean[None], draws]), torch.cat([usage_mean[None], usage[self.ens_draws]])

        def _order(self, z: torch.Tensor, usage: torch.Tensor, member: int | None = None) -> torch.Tensor:
            def o(name: str) -> torch.Tensor:  # an orderer tensor: shared, or this member's
                value = getattr(self, name)
                return value if member is None else value[member]

            projected = (z - o("ord_mean")) @ o("ord_axes").T  # [n, d]
            x = (projected[None] - o("ord_center")[:, None]) / o("ord_spread")[:, None]  # [L, n, d]
            if hasattr(self, "ord_w1"):
                hidden = (torch.einsum("lnd,ldh->lnh", x, o("ord_w1"))[:, :, None] + o("ord_e1")[:, None]
                          + o("ord_b1")[:, None, None])  # [L, n, C, H]
                hidden = hidden * (hidden > 0)
                if hasattr(self, "ord_w2q"):  # quantile_direct: the mean of the expm1 log1p-quantiles
                    raw = torch.einsum("lnch,lhq->lncq", hidden, o("ord_w2q")) + o("ord_bq")[:, None, None]
                    base = raw[..., 0] + o("ord_level")[:, None] + o("ord_b")[:, None, None]
                    steps = torch.nn.functional.softplus(raw[..., 1:])
                    lifts = torch.cat([torch.zeros_like(steps[..., :1]), steps], dim=-1).cumsum(-1)
                    point = torch.expm1(base[..., None] + lifts).mean(-1)
                    net = torch.where(o("ord_seen").bool()[:, None], point, torch.nan)
                else:
                    eta = (torch.einsum("lnch,lch->lnc", hidden, o("ord_w2m")) if hasattr(self, "ord_w2m")
                           else torch.einsum("lnch,lh->lnc", hidden, o("ord_w2")))
                    eta = eta + o("ord_level")[:, None] + o("ord_b")[:, None, None]
                    net = torch.where(o("ord_seen").bool()[:, None], torch.exp(eta), torch.nan)  # [L, n, C]
            else:
                eta = torch.einsum("lnd,ld->ln", x, o("ord_w"))[:, :, None].expand(-1, -1, o("ord_level").shape[1])
                eta = eta + o("ord_level")[:, None] + o("ord_b")[:, None, None]
                net = torch.where(o("ord_seen").bool()[:, None], torch.exp(eta), torch.nan)  # [L, n, C]
            fits = []
            for m in range(usage.shape[0]):  # each fit's own GLM prediction, against its orderer's network and pools
                glm = usage[m].permute(1, 2, 0)  # [C, L, n]
                b = net.permute(2, 0, 1)
                if hasattr(self, "ord_direct"):
                    out = torch.where(torch.isfinite(b) & torch.isfinite(glm), b, glm)
                else:
                    out = glm * order_ratio(glm, b, o("ord_pool_glm"), o("ord_pool_net"), o("ord_pool_n"),
                                            o("ord_alpha"))
                if hasattr(self, "ord_recal_x"):  # the orderer's recalibration per (model, leg)
                    ok = torch.isfinite(out) & (out > 0) & (o("ord_recal_n") > 0)[..., None]
                    lp = torch.log(torch.where(ok, out, torch.ones_like(out)))
                    shifted = torch.exp(lp + _interp(lp, o("ord_recal_x"), o("ord_recal_c"), o("ord_recal_n")))
                    out = torch.where(ok, shifted, out)
                if hasattr(self, "ord_bound"):  # within the factor of the GLM's prediction
                    bound = o("ord_bound")
                    kept = torch.minimum(torch.maximum(out, glm / bound), glm * bound)
                    out = torch.where(torch.isfinite(glm), kept, out)
                fits.append(out.permute(2, 0, 1))  # [n, C, L]
            return torch.stack(fits)

    return StageOne().eval()

