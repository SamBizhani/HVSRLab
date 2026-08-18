"""Background work, without Qt.

Everything slow in this application — scanning a survey, probing a six-week
recording, computing 102 sites — runs through :class:`Job`, so the interface
stays responsive and the same code can be driven from a script.

Deliberately free of Qt imports: the GUI adapts these callbacks into signals
(see ``gui.bridge``), and the batch runner uses them directly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import time
import traceback
from typing import Callable


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


LineHandler = Callable[[str], None]
ProgressHandler = Callable[[float, str], None]
StateHandler = Callable[["Job"], None]


class Cancelled(Exception):
    """Raised inside a job's target when the user asked it to stop."""


@dataclass
class Job:
    """One unit of long-running work.

    The target receives the job itself, so it can log, report progress and
    check for cancellation::

        def work(job):
            for i, site in enumerate(sites):
                job.check_cancel()
                job.progress_to(i / len(sites), site.label())
                job.log_line(f"computed {site.label()}")
    """

    name: str
    target: Callable[["Job"], object] | None = None

    state: JobState = JobState.PENDING
    result: object = None
    error: str = ""
    traceback: str = ""
    progress: float = 0.0
    stage: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    lines: deque = field(default_factory=lambda: deque(maxlen=4000))

    _on_line: list[LineHandler] = field(default_factory=list, repr=False)
    _on_progress: list[ProgressHandler] = field(default_factory=list, repr=False)
    _on_state: list[StateHandler] = field(default_factory=list, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # -- wiring ------------------------------------------------------------
    def on_line(self, handler: LineHandler) -> "Job":
        self._on_line.append(handler)
        return self

    def on_progress(self, handler: ProgressHandler) -> "Job":
        self._on_progress.append(handler)
        return self

    def on_state(self, handler: StateHandler) -> "Job":
        self._on_state.append(handler)
        return self

    # -- reporting ---------------------------------------------------------
    def log_line(self, text: str) -> None:
        self.lines.append(text)
        for handler in self._on_line:
            handler(text)

    def progress_to(self, fraction: float, stage: str = "") -> None:
        self.progress = max(0.0, min(1.0, float(fraction)))
        if stage:
            self.stage = stage
        for handler in self._on_progress:
            handler(self.progress, self.stage)

    def counted(self, done: int, total: int, stage: str = "") -> None:
        self.progress_to(done / total if total else 0.0,
                         f"{stage} ({done}/{total})" if stage else "")

    # -- control -----------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled(f"{self.name} cancelled")

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    # -- execution ---------------------------------------------------------
    def run(self) -> "Job":
        """Run the target in the calling thread."""
        self._set_state(JobState.RUNNING)
        self.started_at = time.time()
        try:
            if self.target is None:
                raise ValueError(f"job {self.name!r} has no target")
            self.result = self.target(self)
            self.state = JobState.CANCELLED if self.cancelled else JobState.SUCCEEDED
        except Cancelled as exc:
            self.state = JobState.CANCELLED
            self.error = str(exc)
        except Exception as exc:                        # noqa: BLE001
            self.state = JobState.FAILED
            self.error = f"{type(exc).__name__}: {exc}"
            self.traceback = traceback.format_exc()
            self.log_line(self.error)
        finally:
            self.finished_at = time.time()
            self._emit_state()
        return self

    def _set_state(self, state: JobState) -> None:
        self.state = state
        self._emit_state()

    def _emit_state(self) -> None:
        for handler in self._on_state:
            handler(self)


class JobQueue:
    """A single worker thread draining a FIFO of jobs.

    One at a time on purpose: these jobs are I/O- and BLAS-bound, and running
    two at once on the same disk makes both slower while making the progress
    display meaningless. Parallelism belongs inside a job (see
    :mod:`hvsrlab.batch`), where it can be sized to the machine.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._current: Job | None = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self.history: list[Job] = []

    @property
    def current(self) -> Job | None:
        return self._current

    @property
    def busy(self) -> bool:
        return self._current is not None

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def submit(self, job: Job) -> Job:
        self._queue.put(job)
        self._ensure_worker()
        return job

    def cancel_current(self) -> None:
        job = self._current
        if job is not None:
            job.cancel()

    def cancel_all(self) -> None:
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            job.state = JobState.CANCELLED
            job._emit_state()
        self.cancel_current()

    def shutdown(self) -> None:
        self._stopping.set()
        self.cancel_all()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._work, name="hvsrlab-jobs",
                                            daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            self._current = job
            try:
                job.run()
            finally:
                self._current = None
                self.history.append(job)
                if len(self.history) > 200:
                    del self.history[:-200]
