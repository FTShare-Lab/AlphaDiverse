import argparse
from dataclasses import replace
import os
from .config import AppConfig
from .agent.loop import run_agent_loop
from .llm import resolve_chat_backend
from .factors.daily import ALPHA20_FEATURES
from .factors.profiles import LEVEL1


def main():
    parser = argparse.ArgumentParser(
        description="AlphaDiverse: iterative alpha factor research"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--membership-data")
    parser.add_argument("--output", default="results/run")
    parser.add_argument(
        "--market",
        choices=["ashare", "hs300", "csi500", "csi1000"],
        default=AppConfig().evaluation.active_markets[0],
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--folds", type=int, default=AppConfig().search_validation.fold_count
    )
    parser.add_argument("--base", choices=["level1", "alpha20"], default="level1")
    parser.add_argument(
        "--base-features",
        nargs="+",
        choices=sorted(set(LEVEL1) | set(ALPHA20_FEATURES)),
        help="Use an explicit Level1/Alpha20 subset instead of --base",
    )
    parser.add_argument("--feature-mask", type=float, default=0)
    parser.add_argument("--inner-start", default="2022-05-01")
    parser.add_argument("--inner-end", default="2025-12-31")
    parser.add_argument("--outer-start", default="2026-01-01")
    parser.add_argument("--outer-end", default="2026-08-31")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.rounds < 1
        or args.folds < 1
        or args.threads < 1
        or (not 0 <= args.feature_mask <= 100)
    ):
        parser.error(
            "rounds/folds/threads must be positive and feature-mask must be between 0 and 100"
        )
    if args.base_features and len(set(args.base_features)) != len(args.base_features):
        parser.error("base-features must be unique")
    for role in ["planner", "realizer", "analysis"]:
        resolve_chat_backend("agent." + role)
    os.environ["ALPHADIVERSE_THREADS"] = str(args.threads)
    cfg = AppConfig()
    cfg = replace(
        cfg,
        search_validation=replace(
            cfg.search_validation,
            fold_count=args.folds,
            min_improved_folds=args.folds // 2 + 1,
            inner_start=args.inner_start,
            inner_end=args.inner_end,
            outer_start=args.outer_start,
            outer_end=args.outer_end,
        ),
        evaluation=replace(
            cfg.evaluation,
            active_markets=(args.market,),
            label_mode=args.market + "_excess",
        ),
        base_features=replace(
            cfg.base_features,
            profile="custom" if args.base_features else args.base,
            features=tuple(args.base_features or ()),
        ),
        agent=replace(cfg.agent, available_feature_mask_pct=args.feature_mask),
    )
    try:
        run_agent_loop(cfg, args.data, args.output, args.rounds, args.membership_data)
    except Exception as exc:
        raise SystemExit(
            "Run failed ("
            + type(exc).__name__
            + f"). Check input data, endpoint configuration, and {args.output}/status.json."
        ) from None


if __name__ == "__main__":
    main()
