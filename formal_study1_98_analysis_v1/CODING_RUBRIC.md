# Frozen Expectation Coding Rubric for Formal Main Analysis

Source: the prior DeepSeek Integrated Study 1 expectation dimension rubric.
This file deliberately preserves its decision rule.  The only output-format
normalization is that the source label `NONE` is written here as `EXP_NONE`,
to match the preregistered formal-analysis label set.  It is not a substantive
category change.

Code the immediate rationale text only.  Do not use or seek participant
metadata, treatment, partner rank, contribution, payoff, Primary Strategy,
session, partner identity, hypotheses, or results.  Code only what the text
explicitly expresses.  Do not infer hidden cognition, latent belief, motive,
or chain-of-thought.

Choose exactly one category:

- `EXP_COOP`: the rationale links the current partner personality cue to an
  expectation that the partner will contribute/leave more, cooperate,
  reciprocate, be trustworthy in a behavioral sense, or be less likely to
  remove from the common fund.
- `EXP_UNCOOP`: the rationale links the current partner personality cue to an
  expectation that the partner will remove more, defect, act selfishly, or be
  less likely to cooperate.
- `EXP_UNCLEAR`: the text links the current personality cue to a
  partner-behavior expectation but the direction is unclear.
- `EXP_NONE`: no personality-linked partner-behavior expectation is expressed.
  A generic statement that a partner might cooperate or defect is `EXP_NONE`
  unless it is linked to the current personality cue.

For every trial return: `blind_trial_id`, `expectation_code`,
`confidence` (`HIGH`, `MEDIUM`, or `LOW`), `supporting_excerpt` (verbatim,
or `NONE`), and `brief_reason`.  Do not add other coding dimensions.

Independence rule: each coder works in a new context and must not see the
other coder's labels, any blind key, or any linked experimental data.
