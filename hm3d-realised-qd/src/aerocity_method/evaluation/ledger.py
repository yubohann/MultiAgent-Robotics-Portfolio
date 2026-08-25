"""Resource caps and intention-to-run failure accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aerocity_method.contracts.io import finite_number, require_identifier
from aerocity_method.contracts.models import ABI_VERSION


class BudgetExceeded(RuntimeError):
    pass


class ResourceLedger:
    def __init__(self, limits: dict[str, float]) -> None:
        if not limits:
            raise ValueError("resource ledger requires limits")
        self.limits = {
            require_identifier(key, "budget key"): finite_number(value, f"limit.{key}")
            for key, value in limits.items()
        }
        if any(value < 0.0 for value in self.limits.values()):
            raise ValueError("resource limits must be non-negative")
        self.used = {key: 0.0 for key in self.limits}

    def consume(self, key: str, amount: float) -> None:
        if key not in self.limits:
            raise ValueError(f"unregistered resource {key!r}")
        resolved = finite_number(amount, f"resource.{key}")
        if resolved < 0.0:
            raise ValueError("resource consumption must be non-negative")
        proposed = self.used[key] + resolved
        if proposed > self.limits[key]:
            raise BudgetExceeded(f"{key} budget would be exceeded")
        self.used[key] = proposed

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ABI_VERSION, "limits": self.limits, "used": self.used}


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    execution_status: str
    scientific_status: str
    retried_from: str | None = None
    failure_class: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if self.execution_status not in {"PLANNED", "EXECUTED", "FAILED", "TIMEOUT", "OOM"}:
            raise ValueError("invalid execution_status")
        if self.scientific_status not in {"NOT_EVALUATED", "VALID", "INVALID"}:
            raise ValueError("invalid scientific_status")
        if self.retried_from is not None:
            require_identifier(self.retried_from, "retried_from")
        if self.failure_class is not None:
            require_identifier(self.failure_class, "failure_class")


class RunLedger:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}

    def add(self, record: RunRecord) -> None:
        if record.run_id in self._records:
            raise ValueError("duplicate run_id")
        if record.retried_from is not None and record.retried_from not in self._records:
            raise ValueError("retry parent must already exist")
        self._records[record.run_id] = record

    def summary(self) -> dict[str, int]:
        records = tuple(self._records.values())
        return {
            "planned": len(records),
            "executed": sum(record.execution_status == "EXECUTED" for record in records),
            "failed": sum(
                record.execution_status in {"FAILED", "TIMEOUT", "OOM"} for record in records
            ),
            "retried": sum(record.retried_from is not None for record in records),
            "scientifically_valid": sum(record.scientific_status == "VALID" for record in records),
        }

    def denominator_complete(self) -> bool:
        summary = self.summary()
        return summary["planned"] == summary["executed"] + summary["failed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ABI_VERSION,
            "records": [asdict(record) for record in self._records.values()],
            "summary": self.summary(),
            "denominator_complete": self.denominator_complete(),
        }
