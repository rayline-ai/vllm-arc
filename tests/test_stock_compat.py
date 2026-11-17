# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The stock engine's lock (``engine.lock``) and its load guard (``compat.require``) on CPU: the lock names the patches
this repository ships and the versions the server asserts; the guard accepts exactly the locked release with the
locked patch set and refuses every other, before vLLM is imported."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from vllm_arc.engine import compat
from vllm_arc.engine import recipe as image
from vllm_arc.server import app as server
from vllm_arc.engine import startup  # noqa: E402

LOCK = compat.read_lock()


def test_the_lock_names_the_fork_patch_release_and_the_asserted_release():
    # The patches come from the vLLM fork's patch-stack release; this repository ships no vLLM code.
    repo, tag, commit = LOCK["patch_source"]
    assert repo == "rayline-ai/vllm" and tag == f"arc-v{LOCK['vllm_version']}-1" and len(commit) == 40
    assert image.PATCHES == tuple(name for _, name in LOCK.patches) == ("arc-v0.33.0.patch",)
    assert not list((ROOT / "vllm_arc").rglob("*.patch"))
    assert (LOCK["vllm_version"], LOCK["vllm_commit"]) == (startup.VLLM_VERSION, startup.VLLM_COMMIT)
    assert LOCK["vllm_wheel_url"].endswith(f"/vllm-{startup.VLLM_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl")
    assert LOCK["python"].startswith(image.PYTHON + ".") and image.BASE_IMAGE.startswith(f"python:{LOCK['python']}-")


def test_the_pins_are_the_full_set_and_exact():
    names = [p.split("==")[0].lower() for p in LOCK.pins]
    assert len(names) == len(set(names))
    assert not {"vllm", "vllm-arc"} & set(names)  # the wheel by URL, the plugin from this repository
    # What the P2 candidate's /health reported (torch, FlashInfer, Triton) and the JIT toolchain it needs.
    for pin in ("torch==2.14.0", "flashinfer-python==0.7.2.post1", "triton==3.8.0", "nvidia-cuda-nvcc==13.0.88",
                "nvidia-cuda-cccl==13.0.85", "ninja==1.13.2"):
        assert pin in LOCK.pins


def test_the_lock_is_package_data():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"vllm_arc.engine" = ["engine.lock"]' in pyproject


@pytest.mark.parametrize("line", ["pin torch", "patch abc vllm_arc/x.patch", "vllm_version 0.33.0 extra",
                                  "uv 0.11.21", "colour blue"])
def test_a_malformed_or_repeated_entry_is_refused(tmp_path, line):
    path = tmp_path / "engine.lock"
    path.write_text(compat.LOCK_PATH.read_text() + line + "\n")
    with pytest.raises(ValueError, match="not a lock entry"):
        compat.read_lock(path)


def test_an_incomplete_lock_is_refused(tmp_path):
    path = tmp_path / "engine.lock"
    path.write_text("".join(line + "\n" for line in compat.LOCK_PATH.read_text().splitlines()
                            if not line.startswith("patched ")))
    with pytest.raises(ValueError, match="incomplete lock"):
        compat.read_lock(path)


def _site(tmp_path: Path, lock: compat.EngineLock, edit: str | None = None, drop: str | None = None) -> Path:
    """A site-packages whose patched files have the lock's hashes: the lock is rewritten to this tree's bytes, so the
    hashes are real ones; ``edit`` / ``drop`` then change or remove one file after the fact."""
    site = tmp_path / "site"
    for i, (_, rel) in enumerate(lock.patched):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_bytes(f"patched file {i}\n".encode())
    if edit:
        (site / edit).write_bytes(b"another patch set\n")
    if drop:
        (site / drop).unlink()
    return site


def _lock_for(tmp_path: Path) -> compat.EngineLock:
    site = _site(tmp_path, LOCK)
    patched = tuple((hashlib.sha256((site / rel).read_bytes()).hexdigest(), rel) for _, rel in LOCK.patched)
    return compat.EngineLock(LOCK.values, LOCK.patches, patched, LOCK.pins)


def test_the_locked_release_with_the_locked_patches_passes(tmp_path):
    lock = _lock_for(tmp_path)
    assert compat.problems(lock, "0.33.0", _site(tmp_path, lock)) == []


def test_another_release_is_refused(tmp_path):
    lock = _lock_for(tmp_path)
    [problem] = compat.problems(lock, "0.34.0", _site(tmp_path, lock))
    assert problem == "vllm 0.34.0 (locked 0.33.0)"
    assert compat.problems(lock, None, None) == ["vllm not installed (locked 0.33.0)", "no importable vllm package"]


@pytest.mark.parametrize("change", ["edit", "drop"])
def test_a_different_or_missing_patch_is_refused(tmp_path, change):
    """Patch 0001 absent, edited, or joined by another patch on one of its files: a patched file's bytes differ."""
    lock = _lock_for(tmp_path)
    rel = lock.patched[1][1]
    [problem] = compat.problems(lock, "0.33.0", _site(tmp_path, lock, **{change: rel}))
    assert problem.startswith(f"{rel}: sha256 ") and ("missing" in problem) == (change == "drop")


def test_require_raises_before_importing_vllm(monkeypatch, tmp_path):
    lock = _lock_for(tmp_path)
    monkeypatch.setattr(compat, "installed_vllm", lambda: ("0.33.0", _site(tmp_path, lock, edit=lock.patched[0][1])))
    with pytest.raises(compat.IncompatibleEngine, match="built for vLLM 0.33.0 with arc-v0.33.0.patch"):
        compat.require(lock)
    monkeypatch.setattr(compat, "installed_vllm", lambda: ("0.33.0", _site(tmp_path, lock)))
    compat.require(lock)


def test_installed_vllm_does_not_import_it(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    compat.installed_vllm()
    assert "vllm" not in sys.modules


def test_the_server_refuses_before_building_an_engine(monkeypatch):
    built = []
    monkeypatch.setattr(server, "build", lambda *a, **k: built.append(True))

    def refuse(lock=None, **_):
        raise compat.IncompatibleEngine("not the locked vLLM")

    monkeypatch.setattr(compat, "require", refuse)
    monkeypatch.setattr(sys, "argv", ["server"])
    with pytest.raises(compat.IncompatibleEngine):
        server.main()
    assert built == []


def _unpatched(tmp_path: Path, lock: compat.EngineLock) -> Path:
    """Stock vLLM of the locked release: none of the lock's patches applied (every patched file differs)."""
    site = _site(tmp_path, lock)
    for _, rel in lock.patched:
        (site / rel).write_bytes(b"stock file\n")
    return site


def test_stock_unpatched_vllm_loads_off_the_retained_route(monkeypatch, tmp_path):
    lock = _lock_for(tmp_path)
    site = _unpatched(tmp_path, lock)
    assert compat.problems(lock, "0.33.0", site, patched=False) == []
    assert len(compat.problems(lock, "0.33.0", site)) == len(lock.patched)
    monkeypatch.setattr(compat, "installed_vllm", lambda: ("0.33.0", site))
    assert compat.require(lock, patched=False, env={}) == []
    with pytest.raises(compat.IncompatibleEngine, match="with arc-v0.33.0.patch"):
        compat.require(lock, patched=True, env={})  # the retained route still needs the locked patches


def test_the_release_is_checked_on_every_route(monkeypatch, tmp_path):
    lock = _lock_for(tmp_path)
    monkeypatch.setattr(compat, "installed_vllm", lambda: ("0.34.0", _site(tmp_path, lock)))
    for patched in (False, True):
        with pytest.raises(compat.IncompatibleEngine, match=r"vllm 0\.34\.0 \(locked 0\.33\.0\)"):
            compat.require(lock, patched=patched, env={})
    assert compat.problems(lock, None, None, patched=False) == ["vllm not installed (locked 0.33.0)"]


def test_the_override_warns_instead_and_returns_what_it_waived(monkeypatch, tmp_path):
    lock = _lock_for(tmp_path)
    monkeypatch.setattr(compat, "installed_vllm", lambda: ("0.33.0", _unpatched(tmp_path, lock)))
    with pytest.raises(compat.IncompatibleEngine, match="VLLM_ARC_UNLOCKED=1"):
        compat.require(lock, patched=True, env={compat.UNLOCKED_ENV: "0"})
    with pytest.warns(RuntimeWarning, match="unqualified"):
        waived = compat.require(lock, patched=True, env={compat.UNLOCKED_ENV: "1"})
    assert len(waived) == len(lock.patched)


def test_the_override_marks_the_engine_id_unqualified(monkeypatch):
    engine = startup.Engine()
    monkeypatch.delenv(compat.UNLOCKED_ENV, raising=False)
    qualified = server.engine_id(engine, "flashinfer", "yarn-f4")
    assert qualified.endswith("+stock-retained") and "unqualified" not in qualified
    monkeypatch.setenv(compat.UNLOCKED_ENV, "1")
    assert server.engine_id(engine, "flashinfer", "yarn-f4") == qualified + "+unqualified"


@pytest.mark.parametrize("env, retained", [({}, False), ({"ARC_STOCK_SESSIONS": "prefix"}, False),
                                           ({"ARC_STOCK_SESSIONS": "retained"}, True)])
def test_the_model_guard_reads_the_route_the_server_exports(env, retained):
    assert compat.retained_route(env) is retained


@pytest.mark.parametrize("argv, env, patched", [
    (["server"], {}, True),
    (["server"], {"ARC_STOCK_SESSIONS": "prefix"}, False),
    (["server", "--eq-addon", "addon"], {}, False),
])
def test_the_server_guards_its_own_route(monkeypatch, argv, env, patched):
    calls = []
    monkeypatch.setattr(compat, "require", lambda lock=None, **k: calls.append(k) or [])
    monkeypatch.setattr(server, "build", lambda *a, **k: (_ for _ in ()).throw(SystemExit("built")))
    monkeypatch.delenv("ARC_STOCK_SESSIONS", raising=False)
    monkeypatch.delenv("ARC_STOCK_PREFIX_UNIT", raising=False)
    monkeypatch.setenv("PATH", server.os.environ.get("PATH", ""))  # main may prepend its venv
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="built"):
        server.main()
    assert calls == [{"patched": patched}]


@pytest.mark.parametrize("sessions", ["retained", "prefix"])
def test_the_app_exports_its_route_for_the_model_guard(monkeypatch, sessions):
    """Whoever builds the app, the engine's processes see its route, so the model guard checks the patched files
    on the retained route even when ``main`` did not run."""
    pytest.importorskip("fastapi")
    monkeypatch.setenv("ARC_STOCK_SESSIONS", "prefix" if sessions == "retained" else "retained")
    server.build(startup.Engine(sessions=sessions), budget=8192, max_model_len=1024, max_sessions=1, idle_seconds=1.0,
                 scratch_slots=1, checkpoint_root="/nonexistent", rope="yarn-f4")
    assert compat.retained_route() is (sessions == "retained")
