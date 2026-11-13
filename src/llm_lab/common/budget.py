"""单进程计算预算账本；估计 GPU 费用，不冒充云平台账单或关机接口。"""

from __future__ import annotations

import math
import time
from pathlib import Path

from .io import read_json, write_json


class BudgetGuard:
    def __init__(self, settings: dict, run_id: str, ledger_path: str | Path):
        self.settings = settings
        self.run_id = run_id
        self.ledger_path = Path(ledger_path)
        self.rate = float(settings.get("hourly_rate_cny") or 0)
        self.run_limit = float(settings.get("run_limit_cny", 50))
        self.project_limit = float(settings.get("project_limit_cny", 500))
        self.reserve = float(settings.get("reserve_cny", 100))
        self.time_limit = float(settings.get("max_runtime_seconds", 1800))
        values = (self.rate, self.run_limit, self.project_limit, self.reserve, self.time_limit)
        if not all(math.isfinite(item) for item in values) or self.rate < 0 or min(self.run_limit, self.project_limit, self.time_limit) <= 0:
            raise ValueError("Budget values must be finite, with positive limits and nonnegative hourly rate")
        if not 0 <= self.reserve < self.project_limit or self.project_limit > 500:
            raise ValueError("Project budget must be <=500 CNY; reserve must leave a positive compute budget")
        ledger = self._read()
        entry = ledger["runs"].get(run_id, {})
        self.prior_cost = float(entry.get("estimated_compute_cny", 0))
        self.prior_seconds = float(entry.get("elapsed_seconds", 0))
        # 不同租赁会话允许不同单价；历史费用直接累加，不按新单价重算。
        self.started = time.monotonic()
        if self.prior_cost >= self.run_limit or self._project_cost(ledger) >= self.project_limit - self.reserve:
            raise ValueError("Recorded run/project compute budget is exhausted")

    def _read(self) -> dict:
        return read_json(self.ledger_path) if self.ledger_path.exists() else {"schema_version": 1, "runs": {}}

    def _project_cost(self, ledger: dict) -> float:
        return sum(float(item["estimated_compute_cny"]) for item in ledger["runs"].values())

    def update(self) -> dict:
        elapsed = time.monotonic() - self.started
        ledger = self._read()
        record = {"elapsed_seconds": self.prior_seconds + elapsed,
                  "estimated_compute_cny": self.prior_cost + elapsed * self.rate / 3600,
                  "hourly_rate_cny": self.rate}
        ledger["runs"][self.run_id] = record
        write_json(self.ledger_path, ledger)
        return {**record, "project_estimated_compute_cny": self._project_cost(ledger)}

    def should_stop(self) -> bool:
        current = self.update()
        return (current["estimated_compute_cny"] >= self.run_limit
                or current["project_estimated_compute_cny"] >= self.project_limit - self.reserve
                or time.monotonic() - self.started >= self.time_limit)

