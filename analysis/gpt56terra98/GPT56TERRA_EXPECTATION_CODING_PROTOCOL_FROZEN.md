# Qwen98 Expectation Coding Protocol — Frozen Before Coding

This document freezes the offline coding protocol for the already-captured Qwen98 `rationale_text`. It authorizes no API call by itself and does not alter any participant or behavioral raw data.

## Scope

The unit is one completed Qwen98 Stage-2 rationale. The coding target is expressed expectation language in the rationale, not a latent belief, true motivation, or internal mental state.

## Taxonomy

Reuse the DeepSeek formal taxonomy exactly:

- `EXP_COOP`: the rationale explicitly expresses an expectation that the partner will contribute/cooperate/leave resources, and that expectation is part of the decision rationale;
- `EXP_UNCOOP`: the rationale explicitly expresses an expectation that the partner will take, defect, contribute little, or otherwise behave uncooperatively, and that expectation is part of the decision rationale;
- `EXP_UNCLEAR`: the rationale explicitly communicates uncertainty/ambivalence about partner behavior or does not allow a reliable directional expectation classification while still discussing the partner/rank cue;
- `EXP_NONE`: the rationale gives a decision reason without an explicit partner-behavior expectation and does not qualify as EXP_UNCLEAR.

Do not merge `EXP_NONE` and `EXP_UNCLEAR`. Do not add a fifth primary category. If a rationale contains multiple clauses, code the explicit expectation content using the frozen DeepSeek mixed-rationale rule and record the supporting span.

## Blinding and inputs

Each coder receives only a randomized blind response ID and the frozen rationale text. The following are prohibited from the coder packet: participant ID, source profile, session, treatment label, partner rank/bin, contribution, payoff, strategy code, personality scores, model-comparison results, hypotheses, and human benchmark percentages.

## Independent coding

Coder 1 and Coder 2 must operate in separate contexts and must not see each other's labels. The same rubric and response text are used. A coder may report `HIGH`, `MEDIUM`, or `LOW` confidence and a short evidence excerpt, but may not request a content-dependent retry.

## Adjudication

Compute raw agreement, Cohen's kappa, and the confusion matrix before any adjudication. All disagreements and all LOW-confidence cases enter adjudication. The adjudicator sees the blind response and the two coder outputs, but not participant metadata or behavioral outcomes. The adjudicator selects one of the four frozen labels and records a brief note; labels are not changed to improve cross-model similarity.

## Integrity rules

- No participant API calls;
- no rationale regeneration;
- no outcome-dependent recoding;
- no deletion of difficult/short responses;
- no exposure to Qwen contribution or strategy results before coding is frozen;
- no human benchmark exposure before coding is frozen;
- raw Qwen rationales remain byte-identical.

This protocol is a transparent adaptation for Qwen98 and is not a claim that Qwen coding reproduces the original human-coder procedure literally.
