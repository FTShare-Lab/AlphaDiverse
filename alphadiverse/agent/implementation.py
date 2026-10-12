import pandas as pd
from ..operators import validate_factor_specs, execute_factor_spec


class ImplementationModule:
    def __init__(self, specification):
        self.specification = specification
        self.available_features = {
            r["name"]: r for r in specification["available_features"]
        }
        self.allowed_layers = tuple(specification["allowed_factor_layers"])
        if not self.available_features or not self.allowed_layers:
            raise ValueError("No available factor inputs")

    def validate_specs(self, raw_specs, *, existing_names=None, expected_count=None):
        return validate_factor_specs(
            raw_specs,
            available_features=self.available_features,
            allowed_layers=self.allowed_layers,
            existing_names=existing_names,
            expected_count=expected_count,
        )

    def implement(self, panel, raw_specs):
        specs = self.validate_specs(raw_specs, existing_names=set(panel.columns))
        computed = {s["name"]: execute_factor_spec(panel, s) for s in specs}
        return (
            pd.concat([panel, pd.DataFrame(computed, index=panel.index)], axis=1),
            specs,
        )
