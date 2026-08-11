"""Safe read-modify-write access to ``fitted_params.json``.

Every fitting script owns one key in a file they all share. The natural way to
write that key is to read the file at the top of ``main()``, work for twenty
minutes, set the key, and write the dict back — which silently discards
anything another script added in between, because the dict in memory predates
it.

That is not hypothetical. On 2026-08-08 the calibration TEST evaluation
clobbered the ``fatigue`` key that ``fit_fatigue.py`` had written while it ran.
It was caught by a spot-check, not by a test, and ``fit_fatigue`` had to be run
again to restore it. ``refit_calibration.py`` and ``fit_fatigue.py`` were fixed
then by re-reading immediately before writing; this module makes that pattern
the only way to write the file, and closes the gap the re-read still leaves.

Two properties, both needed:

* **Re-read inside the lock.** The dict you mutate is read at write time, not
  at the start of your run, so it already contains whatever landed meanwhile.
* **Exclusive lock plus atomic replace.** A re-read alone still races: two
  writers can both read, then both write, and the later one wins. The lock
  serialises the read-modify-write as a unit, and ``os.replace`` means a reader
  never sees a half-written file.

Usage — mutate only your own key, and do it inside the block::

    with fitted.updating() as params:
        params["stage_3"] = {...}

Do not hold the block open across expensive work: everything inside it blocks
every other writer.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from model import constants as C

LOCK_PATH = Path(str(C.FITTED_PARAMS_PATH) + ".lock")

#: json.dumps indent, matched to what the fitting scripts have always written
#: so this change produces no spurious diff.
INDENT = 1


def load() -> dict:
    """Current contents, or ``{}`` if the file does not exist yet."""
    if not C.FITTED_PARAMS_PATH.exists():
        return {}
    return json.loads(C.FITTED_PARAMS_PATH.read_text())


def _write_atomic(params: dict) -> None:
    # Same directory, so os.replace is a rename within one filesystem and
    # therefore atomic: a reader sees either the old file or the new one, never
    # a half-written mixture.
    #
    # The temp name carries the pid. A fixed name would be shared by every
    # writer, so two of them would interleave inside the SAME temp file and
    # replace a half-written one into place — which is what happened when this
    # was tried without the pid, and it produced a torn file rather than a lost
    # key. The lock already serialises writers; this makes the write correct
    # even if it ever does not.
    tmp = C.FITTED_PARAMS_PATH.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(params, indent=INDENT))
        os.replace(tmp, C.FITTED_PARAMS_PATH)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def updating() -> Iterator[dict]:
    """Yield the file's CURRENT contents; write them back on a clean exit.

    The read happens inside the lock, so the yielded dict reflects every write
    that completed before this one started. If the block raises, nothing is
    written — a failed fit must not leave a half-updated parameter file.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            params = load()
            yield params
            _write_atomic(params)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
