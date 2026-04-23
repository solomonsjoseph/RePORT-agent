from __future__ import annotations

import threading
import time
from typing import Any


class GraphRunManager:
    """Background runner for LangGraph progression, keyed by thread_id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def status(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            job = dict(self._jobs.get(thread_id, {}))
        if not job:
            return {"state": "idle", "steps": 0}
        return {
            "state": job.get("state", "idle"),
            "steps": int(job.get("steps", 0)),
            "error": job.get("error"),
            "updated_at": float(job.get("updated_at", 0.0)),
        }

    def is_running(self, thread_id: str) -> bool:
        return self.status(thread_id).get("state") == "running"

    def submit(
        self,
        *,
        thread_id: str,
        app,
        config: dict,
        max_steps: int,
        initial_payload: dict | None = None,
    ) -> bool:
        with self._lock:
            current = self._jobs.get(thread_id)
            if current and current.get("state") in {"running", "error"}:
                return False
            self._jobs[thread_id] = {
                "state": "running",
                "steps": 0,
                "error": None,
                "updated_at": time.time(),
            }

        worker = threading.Thread(
            target=self._run_job,
            kwargs={
                "thread_id": thread_id,
                "app": app,
                "config": config,
                "max_steps": max(1, int(max_steps)),
                "initial_payload": initial_payload,
            },
            daemon=True,
        )
        worker.start()
        return True

    def _run_job(
        self,
        *,
        thread_id: str,
        app,
        config: dict,
        max_steps: int,
        initial_payload: dict | None,
    ) -> None:
        steps = 0
        error = None
        try:
            if initial_payload is not None:
                app.invoke(initial_payload, config=config)
                steps += 1

            while steps < max_steps:
                snapshot = app.get_state(config)
                if not snapshot or not snapshot.next or snapshot.interrupts:
                    break
                app.invoke({}, config=config)
                steps += 1
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._jobs[thread_id] = {
                "state": "error" if error else "done",
                "steps": steps,
                "error": error,
                "updated_at": time.time(),
            }
