from __future__ import annotations

import time
from types import SimpleNamespace

from utils.run_manager import GraphRunManager


class _FakeApp:
    def __init__(self, steps_until_done: int = 2, interrupt_after: int | None = None, invoke_delay: float = 0.0):
        self.steps_until_done = steps_until_done
        self.interrupt_after = interrupt_after
        self.invoke_delay = invoke_delay
        self.invokes: list[dict] = []
        self.calls = 0

    def invoke(self, payload, config=None):
        if self.invoke_delay:
            time.sleep(self.invoke_delay)
        self.invokes.append(payload)
        self.calls += 1
        return {}

    def get_state(self, config=None):
        if self.interrupt_after is not None and self.calls >= self.interrupt_after:
            return SimpleNamespace(next=["human_review_before_run"], interrupts=[{"id": "x"}])
        if self.calls < self.steps_until_done:
            return SimpleNamespace(next=["orchestrator"], interrupts=[])
        return SimpleNamespace(next=[], interrupts=[])


def _wait_done(mgr: GraphRunManager, tid: str, timeout_s: float = 1.5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = mgr.status(tid)
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.02)
    raise AssertionError("background run did not complete in time")


def test_run_manager_submits_and_completes_steps() -> None:
    mgr = GraphRunManager()
    app = _FakeApp(steps_until_done=3)

    started = mgr.submit(
        thread_id="t1",
        app=app,
        config={"configurable": {"thread_id": "t1"}},
        max_steps=5,
        initial_payload={"messages": ["hi"]},
    )
    assert started is True

    st = _wait_done(mgr, "t1")
    assert st["state"] == "done"
    assert st["steps"] == 3
    assert app.invokes[0] == {"messages": ["hi"]}


def test_run_manager_prevents_duplicate_running_job() -> None:
    mgr = GraphRunManager()
    app = _FakeApp(steps_until_done=20, invoke_delay=0.05)

    started1 = mgr.submit(
        thread_id="t2",
        app=app,
        config={"configurable": {"thread_id": "t2"}},
        max_steps=20,
        initial_payload={"messages": ["hi"]},
    )
    started2 = mgr.submit(
        thread_id="t2",
        app=app,
        config={"configurable": {"thread_id": "t2"}},
        max_steps=20,
        initial_payload=None,
    )

    assert started1 is True
    assert started2 is False


def test_run_manager_stops_on_interrupt() -> None:
    mgr = GraphRunManager()
    app = _FakeApp(steps_until_done=10, interrupt_after=1)

    started = mgr.submit(
        thread_id="t3",
        app=app,
        config={"configurable": {"thread_id": "t3"}},
        max_steps=10,
        initial_payload={"messages": ["hi"]},
    )
    assert started is True

    st = _wait_done(mgr, "t3")
    assert st["state"] == "done"
    assert st["steps"] == 1
