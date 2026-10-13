"""Deterministic mapping from the retained 0-100 control scale to levels 1-9."""
from __future__ import annotations

from numbers import Real

LEVEL_BINS = ((0, 11, 1), (12, 22, 2), (23, 33, 3), (34, 44, 4), (45, 55, 5), (56, 66, 6), (67, 77, 7), (78, 88, 8), (89, 100, 9))


def control_score_to_level(score: float) -> int:
    """Map a finite control score using the fixed inclusive boundaries."""
    if not isinstance(score, Real) or isinstance(score, bool):
        raise TypeError("control score must be a numeric value")
    score = float(score)
    if score < 0 or score > 100:
        raise ValueError("control score must be between 0 and 100")
    # The project stores integer scores; for a future continuous score, the
    # interval containing the value remains deterministic.
    for lower, upper, level in LEVEL_BINS:
        if lower <= score <= upper:
            return level
    raise AssertionError(f"unmapped control score: {score}")


def level_to_control_interval(level: int) -> tuple[int, int]:
    if level not in range(1, 10):
        raise ValueError("level must be an integer from 1 to 9")
    lower, upper, _ = LEVEL_BINS[level - 1]
    return lower, upper


def profile_to_levels(profile: dict[str, float], traits=("O", "C", "E", "A", "N")) -> dict[str, int]:
    return {trait: control_score_to_level(profile[trait]) for trait in traits}
