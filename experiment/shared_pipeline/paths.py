"""Canonical path resolution for the unified Guilfoos experiment pipeline.

Every module resolves locations through this file so that the Pilot and the
Formal run read exactly the same frozen Study 0 artefacts.  Nothing here is
run-specific; run-specific output locations are derived from ``run_root()``.
"""
from __future__ import annotations

from pathlib import Path

# .../LLM_BigFive_Personality_Engineering/experiment/shared_pipeline/paths.py
PIPELINE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PIPELINE_DIR.parent                 # .../experiment
PROJECT_DIR = EXPERIMENT_DIR.parent                  # .../LLM_BigFive_Personality_Engineering

# ---- Study 0 (frozen) ------------------------------------------------------
STUDY0_DIR = PROJECT_DIR / "experiment_personality_induction"
STUDY0_SHAPING_DIR = STUDY0_DIR / "personality_shaping"
FROZEN_V3_DIR = STUDY0_SHAPING_DIR / "v3_game_safe"
FROZEN_V3_GENERATOR = FROZEN_V3_DIR / "persona_prompt_generator_v3_game_safe.py"
FROZEN_INTENSITY_QUALIFIERS = STUDY0_SHAPING_DIR / "intensity_qualifiers.json"
FROZEN_V3_RESOURCE = (
    STUDY0_DIR
    / "references"
    / "personality_shaping"
    / "serapio_garcia_2025_trait_adjectives_v3_game_safe.csv"
)
BFI44_ITEMS = STUDY0_DIR / "validation" / "bfi44_items.json"

# ---- Verified human reference population ----------------------------------
FORMAL_POPULATION_DIR = STUDY0_DIR / "formal_population_design"
WOODALL_RAW_DIR = FORMAL_POPULATION_DIR / "data" / "raw"
WOODALL_FILE_1 = WOODALL_RAW_DIR / "woodall_bfi_reversed_scores.sav"
WOODALL_FILE_2 = WOODALL_RAW_DIR / "woodall_bfi_reversed_scores_6_to_8.sav"
WOODALL_PROVENANCE_AUDIT = FORMAL_POPULATION_DIR / "docs" / "woodall_provenance_audit.md"

# ---- Original paper evidence ----------------------------------------------
GUILFOOS_PDF = PROJECT_DIR / "external_reference_not_bundled.pdf"

# ---- This pipeline ---------------------------------------------------------
CONFIG_DIR = EXPERIMENT_DIR / "configs"
PROTOCOL_DIR = EXPERIMENT_DIR / "protocol"
RUNS_DIR = EXPERIMENT_DIR / "runs"
TESTS_DIR = EXPERIMENT_DIR / "tests"
EVIDENCE_DIR = EXPERIMENT_DIR / "evidence"
FREEZE_MANIFEST = EXPERIMENT_DIR / "study0_personality_engineering" / "frozen_v3" / "FREEZE_MANIFEST.json"

# ---- Credentials (never committed, never logged) ---------------------------
ENV_FILE = PROJECT_DIR / "config" / ".env"

TRAITS = ("O", "C", "E", "A", "N")


def env_file() -> Path:
    """Location of the OpenAI-compatible credentials file (never committed)."""
    return ENV_FILE


def run_root(run_type: str, run_id: str | None = None) -> Path:
    """Return the immutable output root for one run."""
    if run_type not in {"pilot", "formal"}:
        raise ValueError(f"run_type must be 'pilot' or 'formal', got {run_type!r}")
    base = RUNS_DIR / run_type
    return base / run_id if run_id else base


def ensure_run_tree(root: Path) -> dict[str, Path]:
    """Create the canonical per-run directory tree and return it."""
    layout = {
        "root": root,
        "config": root / "config",
        "population": root / "population",
        "personas": root / "personas",
        "audit": root / "audit",
        "measurement": root / "measurement",
        "measurement_raw": root / "measurement" / "raw_responses",
        "diagnostics": root / "diagnostics",
        "figures": root / "figures",
        "sessions": root / "sessions",
        "practice": root / "practice",
        "stage2": root / "stage2",
        "stage3": root / "stage3",
        "logs": root / "logs",
        "reports": root / "reports",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
