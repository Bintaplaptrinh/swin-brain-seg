"""Parameter budget shared by the candidate architecture and trainer."""

MAX_PARAMETERS = 10_000_000


def enforce_parameter_budget(model):
    count = sum(p.numel() for p in model.parameters())
    limit = getattr(model, "parameter_limit", MAX_PARAMETERS)
    if count > limit:
        raise ValueError(f"Model has {count:,} parameters; the limit is {limit:,}.")
    return count
