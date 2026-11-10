"""LRS2 exposure-time calculator."""

__all__ = [
    "CHANNELS",
    "SKY_KINDS",
    "CalibrationData",
    "calculate_sn",
    "create_bokeh_app",
    "get_sn",
    "load_calibration",
    "main",
    "save_bokeh",
    "write_sn_table",
]


def __getattr__(name):
    # Keep ``python -m lrs2_etc.calculator`` free of an eager-import warning,
    # while preserving the convenient package-level API.
    if name in __all__:
        from . import calculator

        return getattr(calculator, name)
    raise AttributeError(name)
