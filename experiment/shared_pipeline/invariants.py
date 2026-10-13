"""Program-level invariants for the unified Guilfoos pipeline.

Every invariant below must hold BEFORE any behavioural API call. They are
checked at three points:

1. when a configuration is loaded (:func:`validate_config`);
2. after the population and persona phases (:func:`validate_population_artefacts`);
3. immediately before the first behavioural decision (:func:`validate_pre_behavioral`).

A violation raises ``InvariantViolation`` and stops the run. Nothing is
auto-repaired, re-drawn or re-tuned.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import paths

TRAITS = paths.TRAITS

REQUIRED_SESSION_SIZE = 14
REQUIRED_PERSONA_VERSION = "v3_game_safe"
REQUIRED_MAPPING_SCHEME = "scheme_A_percentile_nine_equal_bins"
REQUIRED_BFI_REPETITIONS = 1
REQUIRED_PRACTICE_ROUNDS = 4
REQUIRED_STAGE2_TREATMENTS = ("A", "B", "C", "D", "E", "F")
REQUIRED_STAGE2_ROUNDS_PER_TREATMENT = 6
REQUIRED_STAGE3_ROUNDS = 6
REQUIRED_REFERENCE_N = 463
RANK_MIN = 1
RANK_MAX = 14
LEVEL_MIN = 1
LEVEL_MAX = 9


class InvariantViolation(AssertionError):
    """A frozen requirement of the experiment pipeline was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantViolation(message)


def validate_config(config: Any) -> list[str]:
    """Assert structural invariants on a loaded ExperimentConfig."""
    checks: list[str] = []
    raw = config.raw

    _require(raw.get("run_type") in {"pilot", "formal"}, "run_type must be 'pilot' or 'formal'")
    checks.append("run_type in {pilot, formal}")

    n_sessions = config.n_sessions
    session_size = config.session_size
    sample_size = config.sample_size
    _require(session_size == REQUIRED_SESSION_SIZE, f"session_size must be {REQUIRED_SESSION_SIZE}, got {session_size}")
    _require(
        sample_size == n_sessions * session_size,
        f"sample_size ({sample_size}) must equal n_sessions ({n_sessions}) x session_size ({session_size})",
    )
    checks.append("sample_size == n_sessions * session_size")
    checks.append(f"session_size == {REQUIRED_SESSION_SIZE}")

    if config.run_type == "pilot":
        _require(n_sessions == 2, f"pilot n_sessions must be 2, got {n_sessions}")
    if config.run_type == "formal":
        _require(n_sessions == 7, f"formal n_sessions must be 7, got {n_sessions}")
    checks.append("run_type-specific n_sessions")

    _require(
        config.persona_version == REQUIRED_PERSONA_VERSION,
        f"persona_version must be {REQUIRED_PERSONA_VERSION!r}, got {config.persona_version!r}",
    )
    _require(
        config.mapping_scheme == REQUIRED_MAPPING_SCHEME,
        f"mapping_scheme must be {REQUIRED_MAPPING_SCHEME!r}, got {config.mapping_scheme!r}",
    )
    checks.append(f"persona_version == {REQUIRED_PERSONA_VERSION}")
    checks.append(f"mapping_scheme == {REQUIRED_MAPPING_SCHEME}")

    _require(
        config.bfi_repetitions == REQUIRED_BFI_REPETITIONS,
        f"bfi.repetitions must be {REQUIRED_BFI_REPETITIONS} for a run-specific measurement, got {config.bfi_repetitions}",
    )
    checks.append(f"bfi.repetitions == {REQUIRED_BFI_REPETITIONS}")

    _require(
        config.practice_rounds == REQUIRED_PRACTICE_ROUNDS,
        f"practice_rounds must be {REQUIRED_PRACTICE_ROUNDS}, got {config.practice_rounds}",
    )
    _require(
        config.stage2_treatments == REQUIRED_STAGE2_TREATMENTS,
        f"stage2_treatments must be {REQUIRED_STAGE2_TREATMENTS}, got {config.stage2_treatments}",
    )
    _require(
        config.stage2_rounds_per_treatment == REQUIRED_STAGE2_ROUNDS_PER_TREATMENT,
        f"stage2_rounds_per_treatment must be {REQUIRED_STAGE2_ROUNDS_PER_TREATMENT}, got {config.stage2_rounds_per_treatment}",
    )
    _require(
        config.stage3_rounds == REQUIRED_STAGE3_ROUNDS,
        f"stage3_rounds must be {REQUIRED_STAGE3_ROUNDS}, got {config.stage3_rounds}",
    )
    checks.append(f"practice_rounds == {REQUIRED_PRACTICE_ROUNDS}")
    checks.append(f"stage2_treatments == {REQUIRED_STAGE2_TREATMENTS}")
    checks.append(f"stage2_rounds_per_treatment == {REQUIRED_STAGE2_ROUNDS_PER_TREATMENT}")
    checks.append(f"stage3_rounds == {REQUIRED_STAGE3_ROUNDS}")

    population = raw.get("reference_population", {})
    _require(
        population.get("name") == "woodall_combined_bfi44",
        f"reference_population.name must be 'woodall_combined_bfi44', got {population.get('name')!r}",
    )
    _require(
        int(population.get("n_complete_profiles", 0)) == REQUIRED_REFERENCE_N,
        f"reference_population.n_complete_profiles must be {REQUIRED_REFERENCE_N}",
    )
    checks.append(f"reference_population == woodall_combined_bfi44 (n={REQUIRED_REFERENCE_N})")

    game = config.game
    _require(int(game["common_fund_units"]) == 20, "common_fund_units must be 20 (Appendix A, p. 127)")
    _require(int(game["max_removal_units"]) == 10, "max_removal_units must be 10 (Appendix A, p. 127)")
    _require(float(game["private_return_per_unit"]) == 0.75, "private_return_per_unit must be 0.75")
    _require(float(game["common_return_per_unit"]) == 0.50, "common_return_per_unit must be 0.50")
    checks.append("removal-frame payoff constants == original")

    _require(
        raw["behavioral"].get("behavioral_frame") == "removal_from_common_fund",
        "behavioral_frame must be 'removal_from_common_fund'; the contribution frame is analysis-only",
    )
    checks.append("behavioral_frame == removal_from_common_fund")

    _require(
        config.memory_rule == "sanitized_repeated_one_shot",
        f"memory_rule must be 'sanitized_repeated_one_shot', got {config.memory_rule!r}",
    )
    checks.append("memory_rule == sanitized_repeated_one_shot")

    required_seeds = {
        "population",
        "session_assignment",
        "tie_break",
        "treatment_order",
        "matching",
    }
    _require(required_seeds.issubset(set(config.seeds)), f"missing seeds: {required_seeds - set(config.seeds)}")
    checks.append("all five run seeds present")

    _require(int(config.retry_limit) >= 1, "retry_limit must be >= 1")
    checks.append("retry_limit >= 1")

    return checks


def validate_levels(levels: Iterable[Sequence[int] | dict[str, int]]) -> None:
    """Assert every persona level vector is EXACTLY five integers in 1..9.

    A short sequence would otherwise zip against ``TRAITS`` and silently drop
    the missing trait, so the length is checked before the values.
    """
    for entry in levels:
        if isinstance(entry, dict):
            _require(set(entry) == set(TRAITS), f"level vector must cover {TRAITS}, got {sorted(entry)}")
            values = {trait: entry[trait] for trait in TRAITS}
        else:
            _require(
                len(entry) == len(TRAITS),
                f"level vector must have exactly {len(TRAITS)} entries for {TRAITS}, got {len(entry)}",
            )
            values = dict(zip(TRAITS, entry))
        for trait, value in values.items():
            _require(isinstance(value, int) and not isinstance(value, bool), f"level {trait} must be an int, got {value!r}")
            _require(LEVEL_MIN <= value <= LEVEL_MAX, f"level {trait}={value} outside {LEVEL_MIN}..{LEVEL_MAX}")


def validate_ranks(ranks: Iterable[int], session_size: int = REQUIRED_SESSION_SIZE) -> None:
    """Assert a within-session rank vector is exactly the permutation 1..14."""
    values = sorted(int(value) for value in ranks)
    _require(
        values == list(range(RANK_MIN, session_size + 1)),
        f"ranks must be a permutation of {RANK_MIN}..{session_size}, got {values}",
    )


def validate_contamination(c2_direct: int) -> None:
    """C2_DIRECT must be exactly zero on the actual generated prompts."""
    _require(
        int(c2_direct) == 0,
        f"C2_DIRECT contamination detected on actual run prompts: {c2_direct} hit(s). ENTIRE RUN STOPS.",
    )


def validate_population_artefacts(
    n_reference: int,
    n_drawn: int,
    expected_sample_size: int,
    unique_source_rows: int,
    c2_direct: int,
) -> list[str]:
    checks: list[str] = []
    _require(n_reference == REQUIRED_REFERENCE_N, f"reference population must be {REQUIRED_REFERENCE_N}, got {n_reference}")
    _require(n_drawn == expected_sample_size, f"drew {n_drawn} profiles but sample_size is {expected_sample_size}")
    _require(unique_source_rows == n_drawn, "sampling must be without replacement: duplicate source rows detected")
    checks.append(f"reference n == {REQUIRED_REFERENCE_N}")
    checks.append(f"drawn n == sample_size == {expected_sample_size}")
    checks.append("whole-row sampling without replacement")
    validate_contamination(c2_direct)
    checks.append("C2_DIRECT == 0 on actual prompts")
    return checks


def validate_pre_behavioral(
    config: Any,
    n_participants: int,
    n_with_persona: int,
    n_with_measurement: int,
    c2_direct: int,
    sessions: dict[str, list[str]],
) -> list[str]:
    """Final gate executed immediately before the first behavioural API call."""
    checks = validate_config(config)
    _require(n_participants == config.sample_size, f"participant count {n_participants} != sample_size {config.sample_size}")
    _require(n_with_persona == n_participants, "every participant must have a frozen persona prompt before behaviour")
    _require(
        n_with_measurement == n_participants,
        "every participant must have exactly one completed BFI-44 administration before behavioral execution; session membership was frozen earlier",
    )
    _require(len(sessions) == config.n_sessions, f"expected {config.n_sessions} sessions, got {len(sessions)}")
    for session_id, members in sessions.items():
        _require(len(members) == config.session_size, f"session {session_id} has {len(members)} members, expected {config.session_size}")
        _require(len(set(members)) == len(members), f"session {session_id} contains duplicate members")
    all_members = [member for members in sessions.values() for member in members]
    _require(len(set(all_members)) == len(all_members), "a participant appears in more than one session")
    validate_contamination(c2_direct)
    checks.append("participant/persona/measurement counts complete")
    checks.append("session sizes exact and mutually exclusive")
    return checks


def require_stage3_resolved(config: Any) -> None:
    """Stage 3 may not execute while its protocol interpretation is unresolved."""
    if config.stage3_enabled is None:
        raise InvariantViolation(
            "Stage 3 is BLOCKED. The original paper contradicts itself on the "
            "Stage-3 sample (methods imply 98 x 6 = 588 trait choices; Table 7 "
            "reports 252 = 3 sessions x 14 x 6). Set behavioral.stage3_enabled "
            "to true or false ONLY after a human reviewer has frozen the "
            "interpretation recorded in protocol/stage3_resolution_dossier.md."
        )
