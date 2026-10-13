"""Load the explicitly labelled qualifier adaptation."""
from __future__ import annotations

import json
from pathlib import Path


def load_intensity_qualifiers(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().with_name("intensity_qualifiers.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "directly_derived_from_supplementary_note_A.7.1":
        raise ValueError("qualifier resource provenance is not the verified supplementary method")
    return data


def qualifier_for_level(level: int, path: Path | None = None) -> dict[str, str]:
    if level not in range(1, 10):
        raise ValueError("level must be from 1 to 9")
    return load_intensity_qualifiers(path)["levels"][str(level)]
