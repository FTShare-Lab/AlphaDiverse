from .planner import PlannerModule
from .retrieval import CapabilityRetriever
from .realizer import RealizerModule


class ResearchPortfolioPolicy:
    def __init__(self):
        (self.planner, self.realizer) = (PlannerModule(), RealizerModule())

    def propose(
        self,
        specification,
        history,
        current_round,
        max_rounds,
        model_sota,
        retained_cards,
        existing_names,
    ):
        planned = self.planner.plan(
            specification,
            history,
            plans_per_round=4,
            current_round=current_round,
            max_rounds=max_rounds,
            model_sota=model_sota,
            model_retained_factors=retained_cards,
        )
        plans = planned["research_plans"]
        cards = CapabilityRetriever(specification).retrieve(plans)
        realized = self.realizer.realize(
            plans,
            cards,
            specification,
            factor_specs_per_plan=2,
            current_round=current_round,
            max_rounds=max_rounds,
            existing_factor_names=existing_names,
        )
        specs = {r["plan_id"]: r["factor_specs"] for r in realized["realizations"]}
        return {
            "research_plans": [
                {**p, "factor_specs": specs[p["plan_id"]]} for p in plans
            ]
        }
