"""PHASE 1 / STEP 4: empirical percentile to prompt Level 1-9 (frozen Scheme A).

Scheme A is the frozen mapping selected during the formal-population design
phase. It divides the percentile axis into nine equal-width regions and is
reproduced here verbatim from
``formal_population_design/analysis/mapping_and_discretisation.py``:

    prompt_level = clamp(ceil(percentile / (100 / 9)), 1, 9)

Two routes are explicitly FORBIDDEN and are not implemented:

* the legacy ``control_score_to_level`` 0-100 bin route (Scheme B), whose bins
  are unequal in percentile terms and leave unmapped gaps;
* passing a human raw score directly into the persona generator.

Only integer Levels 1-9 ever reach ``build_prompt_from_levels``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import paths

TRAITS = paths.TRAITS

SCHEME_A_NAME = "scheme_A_percentile_nine_equal_bins"
NINE_BIN_WIDTH = 100.0 / 9.0
LEVEL_MIN = 1
LEVEL_MAX = 9

# Closed-form boundaries of the nine equal percentile regions, recorded so a
# reviewer can check the bin edges without running code. Right-closed:
# level k covers percentile in ((k-1)*100/9, k*100/9].
SCHEME_A_BOUNDARIES: tuple[tuple[int, float, float], ...] = tuple(
    (level, (level - 1) * NINE_BIN_WIDTH, level * NINE_BIN_WIDTH) for level in range(1, 10)
)


class MappingError(ValueError):
    """A percentile or level fell outside the frozen Scheme A contract."""


def percentile_to_level(percentile: float) -> int:
    """Frozen Scheme A: nine equal percentile regions mapped to Levels 1-9."""
    value = float(percentile)
    if not np.isfinite(value):
        raise MappingError(f"percentile must be finite, got {percentile!r}")
    if value < 0.0 or value > 100.0:
        raise MappingError(f"percentile must lie in [0, 100], got {value}")
    level = int(np.ceil(value / NINE_BIN_WIDTH))
    return int(min(max(level, LEVEL_MIN), LEVEL_MAX))


def level_percentile_interval(level: int) -> tuple[float, float]:
    """Return the (lower, upper] percentile interval a level represents."""
    if level not in range(LEVEL_MIN, LEVEL_MAX + 1):
        raise MappingError(f"level must be an integer {LEVEL_MIN}..{LEVEL_MAX}, got {level!r}")
    for candidate, lower, upper in SCHEME_A_BOUNDARIES:
        if candidate == level:
            return lower, upper
    raise MappingError(f"unmapped level {level}")


def profile_percentiles_to_levels(percentiles: dict[str, float]) -> dict[str, int]:
    """Map a full five-trait percentile profile to integer Levels 1-9."""
    if set(percentiles) != set(TRAITS):
        raise MappingError(f"percentile profile must cover exactly {TRAITS}, got {sorted(percentiles)}")
    return {trait: percentile_to_level(percentiles[trait]) for trait in TRAITS}


def add_prompt_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``prompt_level_<trait>`` columns to a drawn-sample frame."""
    out = frame.copy()
    for trait in TRAITS:
        column = f"human_percentile_{trait}"
        if column not in out.columns:
            raise MappingError(f"missing percentile column {column}; percentiles must be computed against all 463 rows first")
        out[f"prompt_level_{trait}"] = [percentile_to_level(value) for value in out[column]]
    return out


def level_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts per level per trait for the drawn sample (descriptive only)."""
    rows = []
    for trait in TRAITS:
        counts = frame[f"prompt_level_{trait}"].value_counts().reindex(range(LEVEL_MIN, LEVEL_MAX + 1)).fillna(0)
        for level in range(LEVEL_MIN, LEVEL_MAX + 1):
            rows.append(
                {
                    "trait": trait,
                    "level": level,
                    "percentile_lower_exclusive": round((level - 1) * NINE_BIN_WIDTH, 4),
                    "percentile_upper_inclusive": round(level * NINE_BIN_WIDTH, 4),
                    "n": int(counts.loc[level]),
                    "share": float(counts.loc[level]) / float(len(frame)) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def distinct_level_profiles(frame: pd.DataFrame) -> dict[str, int]:
    columns = [f"prompt_level_{trait}" for trait in TRAITS]
    keys = frame[columns].astype(int).apply(lambda row: "|".join(str(int(value)) for value in row), axis=1)
    return {"n_participants": int(len(frame)), "n_distinct_level_profiles": int(keys.nunique()), "n_duplicate_level_profiles": int(len(frame) - keys.nunique())}


def assert_scheme_a_monotonic(values: Iterable[tuple[float, int]]) -> None:
    """Sanity check: Scheme A must be weakly monotonic in the percentile."""
    ordered = sorted(values, key=lambda pair: pair[0])
    previous_level = 0
    for percentile, level in ordered:
        if level < previous_level:
            raise MappingError(f"Scheme A is not monotonic at percentile {percentile}: {level} < {previous_level}")
        previous_level = level
