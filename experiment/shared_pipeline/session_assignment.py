"""PHASE 3: random session assignment before BFI-44 measurement.

Assignment happens once after population/persona construction and before any
measured BFI score exists. Membership is never rebalanced or redrawn from BFI.

Locked rules
------------
* one uniform random permutation of participant IDs, generated from
  ``seeds.session_assignment``;
* the permutation is cut into consecutive blocks of ``session_size``;
* NO personality stratification, NO balancing of session trait means, NO
  control of level proportions per session, NO regrouping to make ranks look
  more even;
* the same function is used by the Pilot (28 -> 2 x 14) and the Formal run
  (98 -> 7 x 14); only the seed and ``n_sessions`` differ.

The original study also assigned 14 subjects per session and computed ranks
within the session, which is why session membership is fixed before ranking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .hashing import provenance_record, sha256_text, write_csv, write_json


class SessionAssignmentError(RuntimeError):
    """Session assignment violated a locked rule."""


def session_label(index: int, run_type: str) -> str:
    prefix = "S" if run_type == "formal" else "P"
    return f"{prefix}{index + 1}"


def assign_sessions(participant_ids: Sequence[str], n_sessions: int, session_size: int, seed: int, run_type: str = "pilot") -> dict[str, list[str]]:
    """Uniform random assignment of participants to fixed-size sessions."""
    ids = list(participant_ids)
    if len(ids) != n_sessions * session_size:
        raise SessionAssignmentError(
            f"expected {n_sessions * session_size} participants for {n_sessions} sessions of {session_size}, got {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise SessionAssignmentError("duplicate participant IDs cannot be assigned to sessions")

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(ids))
    ordered = [ids[int(position)] for position in permutation]

    sessions: dict[str, list[str]] = {}
    for index in range(n_sessions):
        block = ordered[index * session_size : (index + 1) * session_size]
        sessions[session_label(index, run_type)] = block
    return sessions


def assignment_audit(participant_ids: Sequence[str], sessions: dict[str, list[str]], seed: int, n_sessions: int, session_size: int) -> dict[str, Any]:
    ordered = [member for members in sessions.values() for member in members]
    return {
        "seed": int(seed),
        "rng": "numpy.random.Generator(PCG64) via numpy.random.default_rng(seed)",
        "rng_call": f"Generator.permutation({len(participant_ids)}) - exactly once",
        "n_participants": int(len(participant_ids)),
        "n_sessions": int(n_sessions),
        "session_size": int(session_size),
        "session_labels": list(sessions),
        "session_members": {key: list(value) for key, value in sessions.items()},
        "permutation_order": ordered,
        "permutation_sha256": sha256_text("|".join(ordered)),
        "stratification": "NONE",
        "personality_balancing": "NONE",
        "session_trait_mean_control": "NONE",
        "level_proportion_control": "NONE",
        "regrouping": "NONE",
        "assignment_stage": "before BFI-44 measurement; membership frozen before measured scores exist",
        "bfi_can_change_assignment": False,
    }


def session_frame(sessions: dict[str, list[str]], frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``session_id`` and ``session_position`` to a participant frame."""
    lookup = {}
    for session_id, members in sessions.items():
        for position, member in enumerate(members, start=1):
            lookup[member] = (session_id, position)
    out = frame.copy()
    missing = [participant for participant in out["participant_id"] if participant not in lookup]
    if missing:
        raise SessionAssignmentError(f"participants without a session: {missing}")
    out["session_id"] = out["participant_id"].map(lambda participant: lookup[participant][0])
    out["session_position"] = out["participant_id"].map(lambda participant: lookup[participant][1])
    return out


def write_session_artefacts(
    sessions: dict[str, list[str]],
    participant_ids: Sequence[str],
    destination: Path,
    run_type: str,
    seed: int,
    n_sessions: int,
    session_size: int,
    frame: pd.DataFrame | None = None,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        {"session_id": session_id, "session_position": position, "participant_id": member}
        for session_id, members in sessions.items()
        for position, member in enumerate(members, start=1)
    ]
    roster_path = write_csv(destination / f"{run_type}_session_roster.csv", rows, ["session_id", "session_position", "participant_id"])
    audit_path = write_json(
        destination / f"{run_type}_session_assignment_audit.json",
        provenance_record(**assignment_audit(participant_ids, sessions, seed, n_sessions, session_size)),
    )
    written = {"roster_csv": str(roster_path), "audit_json": str(audit_path)}
    if frame is not None:
        enriched = session_frame(sessions, frame)
        enriched_path = destination / f"{run_type}_participants_with_sessions.csv"
        enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
        written["participants_csv"] = str(enriched_path)
    return written
