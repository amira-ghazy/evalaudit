"""A small report object: holds the metrics, flags concerns, prints cleanly."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Check:
    name: str
    value: Optional[float]
    flagged: bool
    note: str = ""


@dataclass
class AuditReport:
    checks: list = field(default_factory=list)

    def add(self, name, value, flagged, note=""):
        self.checks.append(Check(name, value, flagged, note))

    @property
    def passed(self) -> bool:
        return not any(c.flagged for c in self.checks)

    def to_dict(self) -> dict:
        return {c.name: {"value": c.value, "flagged": c.flagged, "note": c.note} for c in self.checks}

    def __str__(self) -> str:
        lines = ["EVALAUDIT REPORT", "=" * 56]
        for c in self.checks:
            val = "  —  " if c.value is None else f"{c.value:+.3f}"
            chip = "FLAG" if c.flagged else " ok "
            lines.append(f"[{chip}] {c.name:<26} {val:>8}   {c.note}")
        lines.append("-" * 56)
        verdict = "PASSES — no measurement red flags" if self.passed \
            else f"NEEDS REVIEW — {sum(c.flagged for c in self.checks)} flag(s)"
        lines.append(verdict)
        return "\n".join(lines)
