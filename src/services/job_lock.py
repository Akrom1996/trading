"""
Host-wide lock for CPU-heavy jobs (model retrains, walk-forward validation).

Every bot container mounts the same data directory as DB_PATH, so an flock on a file there
is shared by all of them: only one heavy job runs at a time and the others queue, leaving
CPU for the trading loops.
"""

import fcntl
import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from config.settings import settings

LOCK_FILE_NAME = "heavy_jobs.lock"


def lock_path(db_path: Optional[str] = None) -> str:
    directory = os.path.dirname(os.path.abspath(db_path or settings.DB_PATH))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, LOCK_FILE_NAME)


@contextmanager
def heavy_job_lock(db_path: Optional[str] = None) -> Iterator[float]:
    """Blocks until no other heavy job holds the lock; yields the seconds spent waiting."""
    with open(lock_path(db_path), "a") as handle:
        started = time.monotonic()
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield time.monotonic() - started
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
