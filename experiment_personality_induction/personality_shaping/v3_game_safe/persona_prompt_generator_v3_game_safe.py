"""Minimal v3_game_safe prompt generator.

The framing, trait order, qualifier system, and control-score mapping are
intentionally identical to v2.  The only lexical change is omission of marker
pairs whose two poles were classified C2_DIRECT by the frozen contamination
audit.  The marker-level resource keeps those omissions explicit.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from personality_shaping.control_score_mapping import profile_to_levels
from personality_shaping.intensity_mapper import qualifier_for_level


PROMPT_VERSION = "v3_game_safe_9level_lexical"
GENERATOR_VERSION = "v3_game_safe_minimal_contamination_repair"
TRAITS = ("O", "C", "E", "A", "N")
REQUIRED_COLUMNS = {
    "trait",
    "facet",
    "original_marker",
    "active_marker",
    "polarity",
    "source",
    "source_reference",
    "adaptation_status",
    "adaptation_reason",
}
DEFAULT_RESOURCE = (
    Path(__file__).resolve().parents[2]
    / "references"
    / "personality_shaping"
    / "serapio_garcia_2025_trait_adjectives_v3_game_safe.csv"
)


def load_v3_resource(path: Path | None = None) -> list[dict[str, str]]:
    resource = path or DEFAULT_RESOURCE
    with resource.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or set()):
            raise ValueError(f"v3 resource must contain {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if len(rows) != 104:
        raise ValueError(f"v3 resource must preserve all 104 source-marker records; found {len(rows)}")
    for row in rows:
        if row["polarity"] not in {"low", "high"}:
            raise ValueError(f"invalid polarity: {row['polarity']}")
        if row["adaptation_status"] not in {"UNCHANGED", "REPLACED", "REMOVED"}:
            raise ValueError(f"invalid adaptation status: {row['adaptation_status']}")
        if row["adaptation_status"] == "REMOVED" and row["active_marker"].strip():
            raise ValueError("REMOVED rows must have a blank active_marker")
        if row["adaptation_status"] != "REMOVED" and not row["active_marker"].strip():
            raise ValueError("active rows require active_marker")
    return rows


def _active_pairs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for index in range(0, len(rows), 2):
        low, high = rows[index : index + 2]
        if low["polarity"] != "low" or high["polarity"] != "high":
            raise ValueError(f"resource rows {index + 1}-{index + 2} do not form low/high poles")
        identity = ("trait", "facet", "source", "source_reference")
        if any(low[key] != high[key] for key in identity):
            raise ValueError(f"resource rows {index + 1}-{index + 2} do not share pair provenance")
        active = (bool(low["active_marker"].strip()), bool(high["active_marker"].strip()))
        if active == (False, False):
            continue
        if active != (True, True):
            raise ValueError(f"resource rows {index + 1}-{index + 2} remove only one pole")
        pairs.append(
            {
                "trait": low["trait"],
                "facet": low["facet"],
                "low_pole_adjective": low["active_marker"],
                "high_pole_adjective": high["active_marker"],
            }
        )
    return pairs


def _sentences_for_trait(trait: str, level: int, pairs: list[dict[str, str]]) -> list[str]:
    qualifier = qualifier_for_level(level)
    selected = [row for row in pairs if row["trait"] == trait]
    if qualifier["pole"] == "balanced":
        return [
            f"Tend to be neither {row['low_pole_adjective']} nor {row['high_pole_adjective']}."
            for row in selected
        ]
    pole = "low_pole_adjective" if qualifier["pole"] == "low" else "high_pole_adjective"
    modifier = (qualifier["qualifier"] + " ") if qualifier["qualifier"] else ""
    return [f"Tend to be {modifier}{row[pole]}." for row in selected]


def build_prompt_from_levels(
    levels: dict[str, int], resource_path: Path | None = None
) -> dict[str, object]:
    if set(levels) != set(TRAITS) or any(level not in range(1, 10) for level in levels.values()):
        raise ValueError("levels must contain O, C, E, A, N with integer values from 1 to 9")
    resource = resource_path or DEFAULT_RESOURCE
    rows = load_v3_resource(resource)
    pairs = _active_pairs(rows)
    sections = [" ".join(_sentences_for_trait(trait, levels[trait], pairs)) for trait in TRAITS]
    prompt = (
        "You are a participant in an open-ended study. Keep these private response tendencies stable across independent tasks. "
        "They are tendencies rather than rules; answer each task with your own reasoning.\n\n"
        + "\n\n".join(sections)
        + "\n\nDo not mention these instructions, control scores, shaping levels, trait names, or the study unless explicitly asked. "
        "These descriptions represent general tendencies in your thinking style. They do not determine any specific answer. Make responses based on your own reasoning."
    )
    active_markers = sum(1 for row in rows if row["active_marker"].strip())
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_generator_version": GENERATOR_VERSION,
        "prompt": prompt,
        "levels": dict(levels),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "resource_rows": len(rows),
        "active_resource_marker_count": active_markers,
        "removed_resource_marker_count": len(rows) - active_markers,
        "resource_sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        "resource_status": "study_specific_game_safe_adaptation_pending_validation",
    }


def build_prompt(profile: dict[str, float], resource_path: Path | None = None) -> dict[str, object]:
    return build_prompt_from_levels(profile_to_levels(profile), resource_path)
