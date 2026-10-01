"""Development-only deterministic fault injection for resilience experiments."""

import os
import time

from src.config import ENVIRONMENT, FAULT_INJECTION_ENABLED


class InjectedFault(RuntimeError):
    pass


class FaultInjector:
    def __init__(self, enabled=None, environment=None, actions=None, sleeper=time.sleep):
        self.enabled = FAULT_INJECTION_ENABLED if enabled is None else enabled
        self.environment = ENVIRONMENT if environment is None else environment
        self.actions = actions or {}
        self.sleeper = sleeper
        if self.enabled and self.environment not in {"development", "test"}:
            raise RuntimeError("Fault injection is permitted only in development or test")

    def action_for(self, stage):
        return self.actions.get(stage, os.getenv(f"FAULT_{stage.upper()}", ""))

    def apply(self, stage):
        if not self.enabled:
            return
        action = self.action_for(stage).strip().lower()
        if not action:
            return
        if action == "fail":
            raise InjectedFault(f"Injected failure: {stage}")
        if action.startswith("delay:"):
            seconds = float(action.split(":", 1)[1])
            if seconds < 0 or seconds > 300:
                raise ValueError("Injected delay must be between 0 and 300 seconds")
            self.sleeper(seconds)
            return
        raise ValueError(f"Unknown injected fault action for {stage}: {action}")

