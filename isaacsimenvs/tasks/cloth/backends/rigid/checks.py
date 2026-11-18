"""Which episode checks need robot/table contact reporting (no simulator imports)."""


def contacts_needed(checks) -> bool:
    """Robot/table contact reporting is required by the press-driven-fold and penetration checks."""
    return bool(getattr(checks, "press_driven_fold", False)
                or float(getattr(checks, "max_robot_penetration_fraction", 0.0)) > 0)
