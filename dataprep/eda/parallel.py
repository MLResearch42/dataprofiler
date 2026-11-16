"""Run per-column work on a thread pool and report progress.

Most of the heavy lifting in EDA (histograms, value counts, quantiles) happens
inside NumPy, pandas and PyArrow, which release the GIL for much of their work,
so a small thread pool gives a real speed-up on wide tables without the cost of
copying data to other processes.

The number of threads can be changed with :func:`set_num_workers` or the
``DATAPREP_NUM_WORKERS`` environment variable (``1`` disables threading).
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

__all__ = ["pmap", "Progress", "set_num_workers", "get_num_workers"]

T = TypeVar("T")
R = TypeVar("R")

_NUM_WORKERS: Optional[int] = None
_ACTIVE: List["Progress"] = []
_ACTIVE_LOCK = threading.Lock()


def set_num_workers(num_workers: Optional[int]) -> None:
    """Set how many threads EDA computations may use.

    Parameters
    ----------
    num_workers
        ``1`` runs everything on the calling thread. ``None`` restores the
        default: the ``DATAPREP_NUM_WORKERS`` environment variable if set,
        otherwise the number of CPUs capped at 8.
    """
    global _NUM_WORKERS  # pylint: disable=global-statement
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    _NUM_WORKERS = num_workers


def get_num_workers() -> int:
    """The number of threads EDA computations may use."""
    if _NUM_WORKERS is not None:
        return _NUM_WORKERS
    env = os.environ.get("DATAPREP_NUM_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 1))


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, NameError):
        return False


def _make_bar(total: int, desc: str) -> Any:
    """Create a tqdm bar: a widget when ipywidgets is usable, otherwise plain text."""
    if _in_notebook():
        try:
            import ipywidgets  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
            from tqdm.notebook import tqdm as tqdm_nb  # pylint: disable=import-outside-toplevel

            return tqdm_nb(total=total, desc=desc, leave=False)
        except Exception:  # pylint: disable=broad-except
            pass  # fall back to the text bar, which works in every frontend
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)


class Progress:
    """A progress bar that only appears if the work takes longer than ``minimum`` seconds.

    Use it as a context manager around a computation. Any :func:`pmap` call made
    inside the block reports to it automatically.

    Parameters
    ----------
    enabled
        Set to False to never show a bar.
    minimum
        Seconds to wait before showing the bar, so quick calls stay silent.
    desc
        Label shown in front of the bar.
    """

    def __init__(self, enabled: bool = True, minimum: float = 1.0, desc: str = "") -> None:
        self.enabled = enabled
        self.minimum = minimum
        self.desc = desc
        self._total = 0
        self._done = 0
        self._bar: Any = None
        self._start = 0.0
        self._lock = threading.Lock()

    def __enter__(self) -> "Progress":
        self._start = time.monotonic()
        with _ACTIVE_LOCK:
            _ACTIVE.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        with _ACTIVE_LOCK:
            if self in _ACTIVE:
                _ACTIVE.remove(self)
        with self._lock:
            if self._bar is not None:
                self._bar.close()
                self._bar = None

    def add_total(self, count: int) -> None:
        """Announce ``count`` more units of work."""
        with self._lock:
            self._total += count
            if self._bar is not None:
                self._bar.total = self._total
                self._bar.refresh()

    def advance(self, count: int = 1, label: Optional[str] = None) -> None:
        """Mark ``count`` units of work as finished."""
        if not self.enabled:
            return
        with self._lock:
            self._done += count
            if self._bar is None:
                if time.monotonic() - self._start < self.minimum:
                    return
                try:
                    self._bar = _make_bar(self._total, self.desc)
                    self._bar.update(self._done)
                except Exception:  # pylint: disable=broad-except
                    self.enabled = False  # never let a progress bar break a computation
                    return
            else:
                self._bar.update(count)
            if label is not None:
                self._bar.set_description(f"{self.desc}: {label}" if self.desc else label, False)


def _active_progress() -> Optional[Progress]:
    with _ACTIVE_LOCK:
        return _ACTIVE[-1] if _ACTIVE else None


def pmap(
    func: Callable[[T], R],
    items: Iterable[T],
    desc: Optional[str] = None,  # pylint: disable=unused-argument
    labels: Optional[Sequence[str]] = None,
) -> List[R]:
    """Apply ``func`` to every item, on a thread pool when that is worthwhile.

    Results are returned in the order of ``items``. The first exception raised by
    ``func`` is re-raised. Progress is reported to the enclosing :class:`Progress`
    block, if there is one.
    """
    items = list(items)
    progress = _active_progress()
    if progress is not None:
        progress.add_total(len(items))

    def run(pos: int) -> R:
        res = func(items[pos])
        if progress is not None:
            progress.advance(1, labels[pos] if labels is not None else None)
        return res

    workers = min(get_num_workers(), len(items))
    # Nested pmap calls run inline: the outer call already uses the pool.
    if workers <= 1 or threading.current_thread().name.startswith("dataprep-eda"):
        return [run(pos) for pos in range(len(items))]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dataprep-eda") as pool:
        return list(pool.map(run, range(len(items))))
