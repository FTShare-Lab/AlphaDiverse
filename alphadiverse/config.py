from dataclasses import dataclass, field


@dataclass(frozen=True)
class DateConfig:
    train_start: str = "2022-05-01"
    train_end: str = "2024-05-31"
    valid_start: str = "2024-06-01"
    valid_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"


@dataclass(frozen=True)
class SearchValidationConfig:
    protocol: str = "inner_multifold"
    inner_start: str = "2022-05-01"
    inner_end: str = "2025-12-31"
    outer_start: str = "2026-01-01"
    outer_end: str = "2026-08-31"
    fold_count: int = 1
    fold_train_ratio: float = 0.7
    fold_valid_ratio: float = 0.1
    fold_test_ratio: float = 0.2
    min_improved_folds: int = 1
    min_sota_delta: float = 0.0
    final_valid_ratio: float = 0.15
    final_refit_scope: str = "all_inner"
    purge_label_overlap: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    label_horizon_days: int = 1
    label_mode: str = "hs300_excess"
    label_return_mode: str = "o2o"
    market_benchmarks: tuple[str, ...] = (
        "hs300:000300.XSHG",
        "csi500:000905.XSHG",
        "csi1000:000852.XSHG",
        "ashare:",
    )
    active_markets: tuple[str, ...] = ("hs300",)
    long_short_quantile: float = 0.2
    min_assets: int = 3
    group_count: int = 5


@dataclass(frozen=True)
class AgentConfig:
    available_feature_mask_pct: float = 70.0
    available_feature_mask_seed: int = 0


@dataclass(frozen=True)
class BaseFeatureConfig:
    profile: str = "level1"
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    dates: DateConfig = field(default_factory=DateConfig)
    search_validation: SearchValidationConfig = field(
        default_factory=SearchValidationConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    base_features: BaseFeatureConfig = field(default_factory=BaseFeatureConfig)
