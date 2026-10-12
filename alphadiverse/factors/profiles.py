LEVEL1 = (
    "ret_1",
    "open_close_ret",
    "high_low_range",
    "close_position",
    "vwap",
    "vwap_deviation",
    "log_turnover",
    "log_volume",
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
    "OPEN0",
    "HIGH0",
    "LOW0",
    "VWAP0",
)


def resolve_base_features(config):
    from .daily import ALPHA20_FEATURES

    if config.features:
        names = list(config.features)
        if len(set(names)) != len(names):
            raise ValueError("Custom base features must be unique")
        unknown = set(names) - (set(LEVEL1) | set(ALPHA20_FEATURES))
        if unknown:
            raise ValueError(
                "Custom base features must come from Level1 or Alpha20: "
                + str(sorted(unknown))
            )
        return names
    if config.profile == "level1":
        return list(LEVEL1)
    if config.profile == "alpha20":
        return list(ALPHA20_FEATURES)
    raise ValueError("Unknown base profile or empty custom feature list")
