"""PHASE 4 / STEP 11: within-session trait ranks 1-14.

Locked rules
------------
* ranks are computed SEPARATELY inside each 14-person session;
* ranking uses ``llm_measured_<trait>`` from this run's single BFI-44
  administration;
* descending: Rank 1 = highest measured trait, Rank 14 = lowest;
* ``human_raw``, ``human_percentile``, ``prompt_level`` and any target control
  score are FORBIDDEN as the rank basis. They are retained in the data for
  analysis only;
* exact ties are broken by a seeded random order (``seeds.tie_break``) and
  every tie group is written to a tie log before the break is applied.

This reproduces the original: "Rankordering refers to the rank of the subject
relative to other subjects in the same session for a given trait" (p. 121),
"1 is the highest rank, 14 is the lowest rank" (Appendix A, p. 127), and
"subjects are not given their own scores or rankings" (p. 123).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import paths
from .hashing import provenance_record, sha256_text, write_csv, write_json
from .invariants import validate_ranks

TRAITS = paths.TRAITS

FORBIDDEN_RANK_BASES = ("human_raw", "human_percentile", "prompt_level", "target_control", "control_score")


class RankingError(RuntimeError):
    """Ranking violated a locked rule."""


def _tie_break_keys(participants: Sequence[str], seed: int) -> dict[str, float]:
    """Deterministic seeded random key per participant for exact-tie breaking."""
    generator = np.random.default_rng(seed)
    values = generator.random(len(participants))
    return {participant: float(value) for participant, value in zip(participants, values)}


def rank_session(
    session_id: str,
    frame: pd.DataFrame,
    trait: str,
    tie_seed: int,
    measured_column: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rank one session on one trait. Returns (ranked frame, tie log entries)."""
    column = measured_column or f"llm_measured_{trait}"
    for forbidden in FORBIDDEN_RANK_BASES:
        if column.startswith(forbidden):
            raise RankingError(f"{column!r} is a forbidden rank basis; ranks must use llm_measured_<trait>")
    if column not in frame.columns:
        raise RankingError(f"missing measured column {column!r} for session {session_id}")

    sub = frame.copy()
    participants = sub["participant_id"].tolist()
    if len(participants) != len(set(participants)):
        raise RankingError(f"session {session_id} contains duplicate participant IDs")
    if sub[column].isna().any():
        raise RankingError(f"session {session_id} has missing measured {trait} values")

    keys = _tie_break_keys(participants, tie_seed)
    sub["_tie_key"] = sub["participant_id"].map(keys)
    sub = sub.sort_values(by=[column, "_tie_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    sub[f"session_rank_{trait}"] = np.arange(1, len(sub) + 1)

    values = sub[column].to_numpy(dtype=float)
    tie_log: list[dict[str, Any]] = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        if end - start > 1:
            group = sub.iloc[start:end]
            tie_log.append(
                {
                    "session_id": session_id,
                    "trait": trait,
                    "tied_value": float(values[start]),
                    "tied_rank_range": f"{start + 1}-{end}",
                    "n_tied": int(end - start),
                    "participants_in_tie_group_ordered_by_seed": "|".join(group["participant_id"].tolist()),
                    "assigned_ranks": "|".join(str(int(value)) for value in group[f"session_rank_{trait}"].tolist()),
                    "tie_break_rule": "seeded random order within the exact-score tie group",
                    "tie_break_seed": int(tie_seed),
                }
            )
        start = end

    validate_ranks(sub[f"session_rank_{trait}"].tolist(), session_size=len(sub))
    return sub[["participant_id", "session_id", column, f"session_rank_{trait}"]].rename(columns={column: f"measured_{trait}"}), tie_log


def rank_all_sessions(frame: pd.DataFrame, tie_seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rank every session on all five traits. Returns (rank table, tie log)."""
    if "session_id" not in frame.columns:
        raise RankingError("frame must already carry session_id; sessions are assigned before ranking")
    rank_tables: list[pd.DataFrame] = []
    tie_log: list[dict[str, Any]] = []
    for session_id, session_frame in frame.groupby("session_id", sort=True):
        merged: pd.DataFrame | None = None
        for trait in TRAITS:
            table, entries = rank_session(str(session_id), session_frame, trait, tie_seed)
            tie_log.extend(entries)
            if merged is None:
                merged = table[["participant_id", "session_id"]]
            merged = merged.merge(table[["participant_id", f"measured_{trait}", f"session_rank_{trait}"]], on="participant_id", how="left", validate="one_to_one")
        rank_tables.append(merged)
    ranks = pd.concat(rank_tables, ignore_index=True)
    for session_id, group in ranks.groupby("session_id"):
        for trait in TRAITS:
            validate_ranks(group[f"session_rank_{trait}"].tolist(), session_size=len(group))
    return ranks, tie_log


def rank_basis_audit() -> dict[str, Any]:
    return {
        "rank_basis": "llm_measured_<trait> from this run's single BFI-44 administration",
        "rank_direction": "descending; Rank 1 = highest measured trait, Rank 14 = lowest",
        "rank_scope": "within each 14-person session, separately per trait",
        "forbidden_bases": list(FORBIDDEN_RANK_BASES),
        "tie_rule": "seeded random order within each exact-score tie group; every tie group logged before the break",
        "own_rank_disclosure": "NEVER shown to the participant (original p. 121, p. 123)",
        "original_source": "Guilfoos & Kurtz (2017) p. 121; Appendix A pp. 127-128; p. 123",
    }


def write_rank_artefacts(
    ranks: pd.DataFrame,
    tie_log: list[dict[str, Any]],
    destination: Path,
    run_type: str,
    tie_seed: int,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    ranks_path = destination / f"{run_type}_session_ranks.csv"
    ranks.to_csv(ranks_path, index=False, encoding="utf-8-sig")
    tie_columns = [
        "session_id",
        "trait",
        "tied_value",
        "tied_rank_range",
        "n_tied",
        "participants_in_tie_group_ordered_by_seed",
        "assigned_ranks",
        "tie_break_rule",
        "tie_break_seed",
    ]
    tie_path = write_csv(destination / f"{run_type}_rank_tie_log.csv", tie_log, tie_columns)
    audit_path = write_json(
        destination / f"{run_type}_ranking_audit.json",
        provenance_record(
            **rank_basis_audit(),
            tie_seed=int(tie_seed),
            n_sessions=int(ranks["session_id"].nunique()),
            n_participants=int(len(ranks)),
            n_tie_groups=int(len(tie_log)),
            ranks_sha256=sha256_text(ranks.to_csv(index=False)),
        ),
    )
    return {"ranks_csv": str(ranks_path), "tie_log_csv": str(tie_path), "audit_json": str(audit_path)}
