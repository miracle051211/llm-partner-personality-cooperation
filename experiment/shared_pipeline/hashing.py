"""Hashing and provenance helpers.

Every artefact that could silently change an experiment result is hashed:
the frozen persona generator, its lexical resource, the BFI-44 key, the raw
human source files, each generated persona prompt, each rendered game prompt
and each configuration snapshot.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def software_versions() -> dict[str, str]:
    """Record the exact software environment for reproducibility."""
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "scipy", "statsmodels", "pyreadstat", "yaml", "matplotlib"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "NOT INSTALLED"
    return versions


def write_json(path: Path | str, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_csv(path: Path | str, rows: list[dict], columns: list[str]) -> Path:
    """Deterministic CSV writer (UTF-8-SIG so Excel opens it correctly)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def provenance_record(**extra: object) -> dict[str, object]:
    """Standard provenance header attached to every generated artefact."""
    record: dict[str, object] = {"generated_utc": utc_now(), "software": software_versions()}
    record.update(extra)
    return record
