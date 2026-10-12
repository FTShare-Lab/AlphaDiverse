from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentValidationConfig:
    correlation_threshold: float = 0.9
    min_coverage: float = 0.2
    min_factor_score: float = 0.05
    min_ic_days: int = 2
