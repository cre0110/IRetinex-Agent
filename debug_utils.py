from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any


def debug_log(scope: str, message: str, **fields: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    suffix = f" | {details}" if details else ""
    print(f"[{timestamp}] [{scope}] {message}{suffix}", flush=True)


class debug_timer:
    def __init__(self, scope: str, message: str, **fields: Any):
        self.scope = scope
        self.message = message
        self.fields = fields
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = perf_counter()
        debug_log(self.scope, f"START {self.message}", **self.fields)
        return self

    def __exit__(self, exc_type, exc, traceback):
        elapsed_ms = int(round((perf_counter() - self.started_at) * 1000))
        if exc is None:
            debug_log(self.scope, f"DONE {self.message}", elapsed_ms=elapsed_ms, **self.fields)
        else:
            debug_log(self.scope, f"FAIL {self.message}", elapsed_ms=elapsed_ms, error=exc, **self.fields)
        return False
