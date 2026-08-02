from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")


def _positive_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


DOWNLOAD_WORKERS = _positive_env("CATCHER_STANCE_DOWNLOAD_WORKERS", 8)
INFERENCE_WORKERS = _positive_env("CATCHER_STANCE_INFERENCE_WORKERS", 1)
REVIEW_WORKERS = _positive_env("CATCHER_STANCE_REVIEW_WORKERS", 1)


class ResourceScheduler:
    def __init__(
        self,
        download_workers: int = DOWNLOAD_WORKERS,
        inference_workers: int = INFERENCE_WORKERS,
        review_workers: int = REVIEW_WORKERS,
    ) -> None:
        self.download_workers = max(1, download_workers)
        self.inference_workers = max(1, inference_workers)
        self.review_workers = max(1, review_workers)
        self._download_slots = threading.BoundedSemaphore(self.download_workers)
        self._condition = threading.Condition()
        self._inference_queue: deque[object] = deque()
        self._active_inference: set[str] = set()
        self._review_pool = ThreadPoolExecutor(
            max_workers=self.review_workers,
            thread_name_prefix="stance-review",
        )
        self._active_downloads = 0
        self._peak_downloads = 0

    @contextmanager
    def download_slot(self) -> Iterator[None]:
        self._download_slots.acquire()
        with self._condition:
            self._active_downloads += 1
            self._peak_downloads = max(self._peak_downloads, self._active_downloads)
        try:
            yield
        finally:
            with self._condition:
                self._active_downloads -= 1
            self._download_slots.release()

    @contextmanager
    def inference_lease(self, run_id: str) -> Iterator[None]:
        token = object()
        with self._condition:
            self._inference_queue.append(token)
            self._condition.wait_for(
                lambda: len(self._active_inference) < self.inference_workers
                and self._inference_queue
                and self._inference_queue[0] is token
            )
            self._inference_queue.popleft()
            self._active_inference.add(run_id)
        try:
            yield
        finally:
            with self._condition:
                self._active_inference.discard(run_id)
                self._condition.notify_all()

    def submit_review(self, function: Callable[..., T], *args, **kwargs) -> Future[T]:
        return self._review_pool.submit(function, *args, **kwargs)

    def stats(self) -> dict[str, int | str | None]:
        with self._condition:
            return {
                "download_workers": self.download_workers,
                "review_workers": self.review_workers,
                "inference_workers": self.inference_workers,
                "active_downloads": self._active_downloads,
                "peak_downloads": self._peak_downloads,
                "queued_inference_runs": len(self._inference_queue),
                "active_inference_run": next(iter(self._active_inference), None),
            }


SCHEDULER = ResourceScheduler()
