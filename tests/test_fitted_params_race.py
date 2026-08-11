"""Concurrent writes to fitted_params.json must not lose a key.

The failure this guards against actually happened: on 2026-08-08 the
calibration TEST evaluation overwrote the ``fatigue`` key that
``fit_fatigue.py`` had written while it was running. Both scripts were
correct in isolation. Nothing crashed, nothing warned, and the loss was found
by a spot-check.

The test runs real processes rather than threads. The GIL would hide the race
these scripts actually run into, which is two interpreters started minutes
apart from a shell or a job runner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from model import constants as C


WRITER = textwrap.dedent("""
    import os, sys, time, random
    from pathlib import Path
    sys.path.insert(0, {repo!r})

    # Redirect the target BEFORE importing model.fitted, which derives its lock
    # path from it. Without this the children would hammer the repo's real
    # fitted_params.json — a test that could damage the thing it is testing.
    from model import constants as C
    C.FITTED_PARAMS_PATH = Path(os.environ["TENNIS_FITTED_PARAMS"])
    from model import fitted as FP
    assert str(FP.LOCK_PATH).startswith(str(C.FITTED_PARAMS_PATH))

    name = sys.argv[1]
    for i in range(25):
        with FP.updating() as params:
            # A slow read-modify-write, which is what a fitting script is: the
            # sleep sits between the read and the write, exactly where the
            # other writer wants to land.
            time.sleep(random.uniform(0.001, 0.004))
            params[name] = {{"i": i, "who": name}}
""")


def _run_writers(tmp_params, names):
    env = {**os.environ, "TENNIS_FITTED_PARAMS": str(tmp_params)}
    procs = [subprocess.Popen([sys.executable, "-c",
                               WRITER.format(repo=str(C.REPO_ROOT)), n],
                              env=env)
             for n in names]
    for p in procs:
        assert p.wait(timeout=120) == 0, p.args
    return json.loads(tmp_params.read_text())


@pytest.fixture
def isolated_params(tmp_path, monkeypatch):
    """Point the module at a scratch file so the real one is never touched."""
    path = tmp_path / "fitted_params.json"
    path.write_text(json.dumps({"stage_1": {"untouched": True}}, indent=1))
    monkeypatch.setattr(C, "FITTED_PARAMS_PATH", path)
    from model import fitted as FP
    monkeypatch.setattr(FP, "LOCK_PATH", tmp_path / "fitted_params.json.lock")
    return path


def test_two_concurrent_writers_lose_nothing(isolated_params):
    """The headline case: two processes, each owning one key, both survive."""
    result = _run_writers(isolated_params, ["writer_a", "writer_b"])

    assert result["writer_a"]["i"] == 24, "writer_a's last write was lost"
    assert result["writer_b"]["i"] == 24, "writer_b's last write was lost"
    assert result["stage_1"] == {"untouched": True}, (
        "a key neither writer owns was dropped")


def test_four_concurrent_writers_lose_nothing(isolated_params):
    names = [f"writer_{i}" for i in range(4)]
    result = _run_writers(isolated_params, names)
    for n in names:
        assert result[n]["i"] == 24, f"{n}'s last write was lost"
    assert result["stage_1"] == {"untouched": True}


def test_a_failed_block_writes_nothing(isolated_params):
    """A fit that raises must not leave a half-updated parameter file."""
    from model import fitted as FP
    before = isolated_params.read_text()
    with pytest.raises(RuntimeError):
        with FP.updating() as params:
            params["stage_9"] = {"half": "written"}
            raise RuntimeError("the fit blew up")
    assert isolated_params.read_text() == before


def test_updating_sees_writes_that_landed_after_the_caller_started(
        isolated_params):
    """The re-read half of the fix, without any concurrency.

    A fitting script reads the file at the top of main(), works, then writes.
    updating() must hand it the CURRENT contents, not a stale snapshot.
    """
    from model import fitted as FP
    stale = FP.load()                      # what a long run would be holding
    with FP.updating() as params:          # something else lands meanwhile
        params["landed_meanwhile"] = True
    assert "landed_meanwhile" not in stale

    with FP.updating() as params:
        params["written_after"] = True
    final = FP.load()
    assert final["landed_meanwhile"] is True
    assert final["written_after"] is True
    assert final["stage_1"] == {"untouched": True}
