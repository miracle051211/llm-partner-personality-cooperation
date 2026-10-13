"""PHASE 5 / STEPS 12-14: treatment order and partner matching.

Original constraints (LOCKED)
-----------------------------
* pool = the other 13 members of the same 14-person session (p. 121-122);
* a new randomly assigned anonymous partner every round (pp. 121, 126-127);
* within a treatment a subject never meets the same partner twice, so six
  rounds use six distinct partners (p. 121; p. 122 fn. 6);
* repeats across treatments are allowed, and with 14 per session a subject
  meets the same partner approximately three times over the experiment
  (p. 122 fn. 6);
* "We use a predetermined random matching system based on the subject's
  computer station in the lab" (p. 122 fn. 6) - the schedule is fixed in
  advance, not drawn live;
* the order of the six treatments is randomised ACROSS SESSIONS, i.e. the
  session is the randomisation unit (Table 2 note, p. 121).

The exact original algorithm is NOT REPORTED. The implementation below is a
declared IMPLEMENTATION ADAPTATION that satisfies every locked constraint:

1. Per session, build the complete round-robin one-factorisation of the 14
   participants (circle method): 13 rounds of 7 disjoint pairs in which every
   one of the 91 possible pairs occurs exactly once.
2. Seed-shuffle the 13 rounds once per session.
3. For each block (practice, each of the six Stage-2 treatments, Stage 3) draw
   WITHOUT replacement the number of rounds that block needs from those 13.
   Because a pair occurs in exactly one of the 13 rounds, drawing distinct
   rounds guarantees six distinct partners inside any six-round block, which is
   the locked within-treatment no-repeat rule.
4. Blocks draw independently, so the same pair can recur across blocks. The
   expected number of encounters per pair over a full 46-round run is
   46 / 13 = 3.54, which reproduces the original's "approximately 3 times".

Everything is generated from ``seeds.matching`` / ``seeds.treatment_order``
through ``numpy.random.SeedSequence`` spawn keys, so the schedule is
deterministic, reproducible and independent of any behavioural result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .hashing import provenance_record, sha256_text, write_csv, write_json

TREATMENTS = ("A", "B", "C", "D", "E", "F")

# Fixed block order used to derive spawn keys. Changing this order changes the
# schedule, so it is part of the frozen algorithm.
BLOCK_ORDER = ("practice", "A", "B", "C", "D", "E", "F", "stage3")


class MatchingError(RuntimeError):
    """A matching or treatment-order constraint was violated."""


# --------------------------------------------------------------------------
# one-factorisation
# --------------------------------------------------------------------------
def one_factorization(participants: Sequence[str]) -> list[list[tuple[str, str]]]:
    """Circle-method round robin: n-1 rounds of n/2 disjoint pairs.

    Every unordered pair of participants appears exactly once.
    """
    ids = list(participants)
    n = len(ids)
    if n < 2 or n % 2 != 0:
        raise MatchingError(f"one-factorisation needs an even number >= 2 of participants, got {n}")
    fixed = ids[0]
    rotating = ids[1:]
    rounds: list[list[tuple[str, str]]] = []
    for round_index in range(n - 1):
        order = [fixed] + rotating
        pairs = [(order[i], order[n - 1 - i]) for i in range(n // 2)]
        rounds.append(sorted((min(a, b), max(a, b)) for a, b in pairs))
        rotating = [rotating[-1]] + rotating[:-1]
    return rounds


def verify_one_factorization(rounds: list[list[tuple[str, str]]], participants: Sequence[str]) -> dict[str, Any]:
    """Prove the schedule is a valid one-factorisation."""
    ids = set(participants)
    n = len(ids)
    seen_pairs: set[tuple[str, str]] = set()
    for index, pairs in enumerate(rounds):
        if len(pairs) != n // 2:
            raise MatchingError(f"round {index + 1} has {len(pairs)} pairs, expected {n // 2}")
        covered: list[str] = []
        for a, b in pairs:
            if a == b:
                raise MatchingError(f"round {index + 1} contains a self-match: {a}")
            if a not in ids or b not in ids:
                raise MatchingError(f"round {index + 1} contains a participant outside the session: {a}, {b}")
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                raise MatchingError(f"pair {key} appears in more than one round of the factorisation")
            seen_pairs.add(key)
            covered += [a, b]
        if len(set(covered)) != n:
            raise MatchingError(f"round {index + 1} does not cover every participant exactly once")
    expected_pairs = n * (n - 1) // 2
    if len(seen_pairs) != expected_pairs:
        raise MatchingError(f"factorisation produced {len(seen_pairs)} distinct pairs, expected {expected_pairs}")
    return {"n_rounds": len(rounds), "n_pairs_per_round": n // 2, "n_distinct_pairs": len(seen_pairs), "valid": True}


def _shuffle_rounds(rounds: list[list[tuple[str, str]]], seed: int, spawn_key: tuple[int, ...]) -> list[list[tuple[str, str]]]:
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=spawn_key)
    generator = np.random.default_rng(sequence)
    order = generator.permutation(len(rounds))
    return [rounds[int(position)] for position in order]


def _draw_block_rounds(n_available: int, n_needed: int, seed: int, spawn_key: tuple[int, ...]) -> list[int]:
    if n_needed > n_available:
        raise MatchingError(f"block needs {n_needed} distinct rounds but only {n_available} exist in the factorisation")
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=spawn_key)
    generator = np.random.default_rng(sequence)
    return [int(value) for value in generator.choice(n_available, size=n_needed, replace=False)]


# --------------------------------------------------------------------------
# treatment order (randomisation unit = SESSION)
# --------------------------------------------------------------------------
def session_treatment_order(session_index: int, seed: int, treatments: Sequence[str] = TREATMENTS) -> list[str]:
    """One seeded uniform permutation of the six treatments for one session.

    The session is the randomisation unit, exactly as in the original. Two
    sessions may receive the same order by chance; that is not corrected.
    """
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=(session_index, 9999))
    generator = np.random.default_rng(sequence)
    order = generator.permutation(len(treatments))
    return [treatments[int(position)] for position in order]


def treatment_orders(session_ids: Sequence[str], seed: int) -> dict[str, list[str]]:
    ordered_sessions = sorted(session_ids)
    return {
        session_id: session_treatment_order(index, seed)
        for index, session_id in enumerate(ordered_sessions)
    }


# --------------------------------------------------------------------------
# per-session matching schedule
# --------------------------------------------------------------------------
def session_matching_schedule(
    session_id: str,
    session_index: int,
    participants: Sequence[str],
    block_round_counts: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Build the complete partner schedule for one session.

    ``block_round_counts`` maps a block name to the number of rounds it needs,
    e.g. ``{"practice": 4, "A": 6, ..., "F": 6, "stage3": 6}``.
    """
    unknown = set(block_round_counts) - set(BLOCK_ORDER)
    if unknown:
        raise MatchingError(f"unknown matching blocks: {sorted(unknown)}")
    ids = list(participants)
    base_rounds = one_factorization(ids)
    factorisation_audit = verify_one_factorization(base_rounds, ids)
    shuffled = _shuffle_rounds(base_rounds, seed, (session_index, 0))

    schedule: dict[str, list[list[tuple[str, str]]]] = {}
    round_indices: dict[str, list[int]] = {}
    for block_index, block in enumerate(BLOCK_ORDER, start=1):
        if block not in block_round_counts:
            continue
        needed = int(block_round_counts[block])
        if needed <= 0:
            continue
        chosen = _draw_block_rounds(len(shuffled), needed, seed, (session_index, block_index))
        round_indices[block] = chosen
        schedule[block] = [shuffled[position] for position in chosen]

    return {
        "session_id": session_id,
        "session_index": session_index,
        "participants": ids,
        "n_participants": len(ids),
        "factorisation": shuffled,
        "factorisation_audit": factorisation_audit,
        "block_round_indices": round_indices,
        "schedule": schedule,
        "seed": int(seed),
        "algorithm": "seeded circle-method one-factorisation; independent per-block draw of distinct rounds without replacement",
        "adaptation_status": "IMPLEMENTATION ADAPTATION - the original station-based generator is NOT REPORTED",
    }


def build_all_schedules(
    sessions: dict[str, Sequence[str]],
    block_round_counts: dict[str, int],
    seed: int,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(sessions)
    return {
        session_id: session_matching_schedule(session_id, index, sessions[session_id], block_round_counts, seed)
        for index, session_id in enumerate(ordered)
    }


def schedule_to_long(schedules: dict[str, dict[str, Any]], sessions: dict[str, Sequence[str]]) -> pd.DataFrame:
    """Flatten schedules into one row per participant-block-round."""
    rows: list[dict[str, Any]] = []
    for session_id, schedule in schedules.items():
        members = list(sessions[session_id])
        for block, block_rounds in schedule["schedule"].items():
            for round_number, pairs in enumerate(block_rounds, start=1):
                for pair_index, (a, b) in enumerate(pairs, start=1):
                    rows.append(
                        {
                            "session_id": session_id,
                            "block": block,
                            "round_in_block": round_number,
                            "pair_index": pair_index,
                            "participant_id": a,
                            "partner_id": b,
                            "factorisation_round_index": schedule["block_round_indices"][block][round_number - 1],
                        }
                    )
                    rows.append(
                        {
                            "session_id": session_id,
                            "block": block,
                            "round_in_block": round_number,
                            "pair_index": pair_index,
                            "participant_id": b,
                            "partner_id": a,
                            "factorisation_round_index": schedule["block_round_indices"][block][round_number - 1],
                        }
                    )
        for member in members:
            if not any(row["participant_id"] == member for row in rows if row["session_id"] == session_id):
                raise MatchingError(f"participant {member} never appears in the schedule of session {session_id}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# validators (must PASS before any behavioural API call)
# --------------------------------------------------------------------------
def validate_schedule(long: pd.DataFrame, sessions: dict[str, Sequence[str]], expected_rounds: dict[str, int]) -> dict[str, Any]:
    """Enforce every locked matching constraint. Raises on the first violation."""
    checks: dict[str, Any] = {}
    session_of = {member: session_id for session_id, members in sessions.items() for member in members}

    # 1. no self-match
    self_matches = long[long["participant_id"] == long["partner_id"]]
    if len(self_matches):
        raise MatchingError(f"self-matches detected: {len(self_matches)} rows")
    checks["no_self_match"] = True

    # 2. no cross-session match
    cross = long[long["participant_id"].map(session_of) != long["partner_id"].map(session_of)]
    if len(cross):
        raise MatchingError(f"cross-session matches detected: {len(cross)} rows")
    checks["no_cross_session_match"] = True

    # 3. symmetry: every (block, round, pair) appears from both viewpoints
    grouped = long.groupby(["session_id", "block", "round_in_block", "pair_index"])["participant_id"].apply(lambda s: tuple(sorted(s)))
    asymmetric = [key for key, value in grouped.items() if len(value) != 2]
    if asymmetric:
        raise MatchingError(f"asymmetric pairs detected: {asymmetric[:5]}")
    checks["pair_symmetry"] = True

    # 4. every participant appears exactly once per block-round
    counts = long.groupby(["session_id", "block", "round_in_block"])["participant_id"].agg(["count", "nunique"])
    bad = counts[counts["count"] != counts["nunique"]]
    if len(bad):
        raise MatchingError(f"a participant appears twice in one round: {bad.head().to_dict('index')}")
    for (session_id, block, _round), row in counts.iterrows():
        expected_members = len(sessions[session_id])
        if int(row["count"]) != expected_members:
            raise MatchingError(f"round {session_id}/{block}/{_round} covers {row['count']} of {expected_members} participants")
    checks["full_coverage_every_round"] = True

    # 5. correct number of rounds per block
    for block, expected in expected_rounds.items():
        observed = long[long["block"] == block].groupby("session_id")["round_in_block"].nunique()
        for session_id, value in observed.items():
            if int(value) != int(expected):
                raise MatchingError(f"session {session_id} block {block} has {value} rounds, expected {expected}")
        missing_sessions = set(sessions) - set(observed.index)
        if expected > 0 and missing_sessions:
            raise MatchingError(f"block {block} is missing sessions: {sorted(missing_sessions)}")
    checks["round_counts_per_block"] = True

    # 6. no repeated partner WITHIN a block
    for (session_id, block), group in long.groupby(["session_id", "block"]):
        per_participant = group.groupby("participant_id")["partner_id"].apply(list)
        for participant, partners in per_participant.items():
            if len(set(partners)) != len(partners):
                raise MatchingError(f"repeated partner inside block {block} for {participant} in session {session_id}")
    checks["no_repeated_partner_within_block"] = True

    # 7. descriptive: encounters per pair across the whole run.
    # ``long`` stores every pair twice per round (once from each viewpoint), so
    # an encounter is one distinct (block, round_in_block) per unordered pair.
    # Counting rows instead would report exactly double the true encounter rate.
    with_pair = long.assign(
        pair=long.apply(lambda row: "|".join(sorted((row["participant_id"], row["partner_id"]))), axis=1)
    )
    encounters = (
        with_pair.drop_duplicates(subset=["session_id", "pair", "block", "round_in_block"])
        .groupby(["session_id", "pair"])
        .size()
    )
    per_session_totals = encounters.groupby(level=0).sum()
    n_distinct_pairs = encounters.groupby(level=0).count()
    # Distinct partners per participant, taken from the full (both-viewpoints)
    # frame so no participant's own row can be dropped by the pair de-dup.
    distinct_partners = with_pair.groupby(["session_id", "participant_id"])["partner_id"].nunique()
    session_size = len(next(iter(sessions.values()))) if sessions else 0
    checks["encounter_descriptives"] = {
        "mean_encounters_per_pair": float(encounters.mean()) if len(encounters) else None,
        "max_encounters_per_pair": int(encounters.max()) if len(encounters) else None,
        "min_encounters_per_pair": int(encounters.min()) if len(encounters) else None,
        "expected_encounters_per_pair": (
            # Each round pairs every participant once, so a session generates
            # n/2 pair-encounters per round over n*(n-1)/2 possible pairs:
            # expected encounters per pair = total_rounds / (n - 1).
            round(sum(expected_rounds.values()) / (session_size - 1), 4)
            if session_size > 1 and expected_rounds
            else None
        ),
        "mean_distinct_partners_per_participant": float(distinct_partners.mean()) if len(distinct_partners) else None,
        "total_rounds_per_participant": int(sum(expected_rounds.values())) if expected_rounds else None,
        "distinct_pairs_used_per_session": {str(key): int(value) for key, value in n_distinct_pairs.items()},
        "total_pair_encounters_per_session": {str(key): int(value) for key, value in per_session_totals.items()},
        "original_reference": "p. 122 fn. 6: approximately 3 encounters with the same partner over the experiment",
        "counting_rule": "one encounter = one distinct (block, round) per unordered pair; the symmetric duplicate row is not counted",
    }
    checks["cross_block_repeats_allowed"] = True
    return checks


def write_matching_artefacts(
    schedules: dict[str, dict[str, Any]],
    long: pd.DataFrame,
    orders: dict[str, list[str]],
    destination: Path,
    run_type: str,
    seed: int,
    treatment_seed: int,
    validation: dict[str, Any],
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    long_path = destination / f"{run_type}_matching_schedule.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")

    order_rows = [
        {"session_id": session_id, "position": position, "treatment": treatment}
        for session_id, order in orders.items()
        for position, treatment in enumerate(order, start=1)
    ]
    order_path = write_csv(destination / f"{run_type}_treatment_order.csv", order_rows, ["session_id", "position", "treatment"])

    audit_path = write_json(
        destination / f"{run_type}_matching_audit.json",
        provenance_record(
            matching_seed=int(seed),
            treatment_order_seed=int(treatment_seed),
            algorithm=schedules[sorted(schedules)[0]]["algorithm"] if schedules else None,
            adaptation_status="IMPLEMENTATION ADAPTATION",
            treatment_order_randomisation_unit="SESSION (original Table 2 note, p. 121)",
            treatment_orders=orders,
            block_round_indices={session_id: schedule["block_round_indices"] for session_id, schedule in schedules.items()},
            factorisation_audit={session_id: schedule["factorisation_audit"] for session_id, schedule in schedules.items()},
            validation=validation,
            schedule_sha256=sha256_text(long.to_csv(index=False)),
        ),
    )
    return {"schedule_csv": str(long_path), "treatment_order_csv": str(order_path), "audit_json": str(audit_path)}
