def __getattr__(name: str):
    if name == "HAS_PYMBAR":
        return has_pymbar()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
