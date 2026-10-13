# Qwen Expectation Coding Protocol Amendment V1 — Frozen Before Execution

## Status and scope

This document is a prospective execution-layer amendment to
`QWEN_EXPECTATION_CODING_PROTOCOL_FROZEN.md`. It completes previously reported
specification gaps before any Qwen expectation category outcome is observed.
It does not change the four-category taxonomy, the frozen Qwen rationale text,
the DeepSeek expectation table, or any behavioral raw file.

The amendment is not outcome-dependent recoding. It is binding for the Qwen98
expectation-coding run only.

## 1. Blind inputs and retained rationale text

Each coder and the adjudicator receive only:

- `blind_id`;
- the frozen `rationale_text` associated with that blind ID.

Personality wording, partner descriptions, and personality-related language
naturally present in the rationale remain byte-for-byte unchanged. No coder or
adjudicator receives injected personality metadata, treatment labels, ranks,
rank bins, participant IDs, source profiles, contribution, removal, payoff,
strategy, session, hypotheses, results, or cross-model information.

## 2. Coder execution configuration

To preserve the original DeepSeek formal expectation implementation, both
independent Qwen coders use the same configuration:

- model: `deepseek-flash`;
- provider: the provider configured in the original DeepSeek expectation
  coding environment;
- transport/API interface: the same OpenAI-compatible interface and endpoint
  configuration used by the original DeepSeek expectation coding script;
- temperature: `0.2`;
- top_p: `1.0`;
- max_tokens: `256`;
- thinking/reasoning: disabled;
- independent context/API call per coder;
- no shared coder output or context state.

The currently effective local provider configuration is read from the existing
project environment file at execution time and is never written to an output
file or log field containing credentials.

## 3. Coder output schema

Each successful coder response must be one strict JSON object with exactly the
following keys and no additional keys:

```json
{
  "blind_id": "...",
  "expectation_code": "EXP_COOP | EXP_UNCOOP | EXP_UNCLEAR | EXP_NONE",
  "confidence": "HIGH | MEDIUM | LOW",
  "supporting_excerpt": "...",
  "brief_reason": "..."
}
```

`blind_id` must exactly match the supplied blind ID. `supporting_excerpt` must
be a verbatim span from the supplied rationale or the literal `NONE` when no
supporting span exists. `brief_reason` may explain only the frozen expectation
classification and may not mention downstream behavior or results.

## 4. Frozen taxonomy and operational rule

Only the following labels are legal:

- `EXP_COOP`: the rationale links the current partner cue to an expectation
  that the partner will contribute/leave more, cooperate, reciprocate, be
  trustworthy behaviorally, or be less likely to remove from the common fund;
- `EXP_UNCOOP`: the rationale links the current partner cue to an expectation
  that the partner will remove more, defect, act selfishly, contribute less, or
  be less likely to cooperate;
- `EXP_UNCLEAR`: the rationale links the current partner cue to partner
  behavior but the directional expectation is ambiguous, conditional,
  uncertain, or contains unresolved cooperative and uncooperative possibilities;
- `EXP_NONE`: no personality-linked expectation about partner contribution or
  cooperation behavior is expressed. Personality description, self-strategy,
  payoff, rules, or generic non-cue-linked statements alone are `EXP_NONE`.

Coders must use explicit rationale evidence and must not infer a behavioral
expectation from trait stereotypes.

## 5. Mixed-rationale precedence rule

1. If the text gives a clear dominant or concluding directional expectation,
   code that final direction.
2. If two directional expectations are equally explicit and no dominant
   conclusion is present, code `EXP_UNCLEAR`.
3. If earlier possibilities are followed by an explicit final expectation, use
   the later final expectation.
4. If the text contains personality inference but no projection to partner
   cooperation/contribution behavior, code `EXP_NONE`.
5. Never resolve mixed content using contribution, strategy, payoff, rank, or
   any information outside the supplied rationale.

## 6. Technical invalid output and bounded retry

An output is technically invalid if it is not valid JSON, has missing or
mismatched `blind_id`, contains an illegal label or confidence value, omits a
required field, returns multiple categories, contains a non-verbatim
supporting excerpt, or is a refusal/unparseable response.

Each coder/adjudicator logical call allows at most **three technical retries**
(maximum four total attempts including the initial attempt). Retries are
allowed only for API/network error, timeout, invalid JSON, schema violation,
truncation, or missing required fields. A valid but unexpected classification,
LOW confidence, coder disagreement, or a result that is inconvenient for the
paper never triggers a retry.

If all four attempts fail, retain a `CODER_TECHNICAL_FAILURE` record and send
the blind ID to the adjudication/error-resolution queue. No classification is
guessed.

## 7. Adjudication trigger and blindness

Adjudication is required when coder labels disagree, either coder reports LOW
confidence, a technical failure occurs, or a schema-valid response has
conflicting evidence.

The adjudicator sees only the blind rationale and the two coder outputs. The
adjudicator does not see any behavioral or participant metadata and performs an
independent application of this amendment.

## 8. Adjudicator output schema

Each adjudication response must be strict JSON with exactly these keys:

```json
{
  "blind_id": "...",
  "final_expectation_code": "EXP_COOP | EXP_UNCOOP | EXP_UNCLEAR | EXP_NONE",
  "final_confidence": "HIGH | MEDIUM | LOW",
  "decision_basis": "...",
  "used_coder1": true,
  "used_coder2": true
}
```

The adjudicator must independently decide from the rationale and taxonomy. It
must not default to Coder 1, Coder 2, or majority. If a single direction cannot
be determined, use `EXP_UNCLEAR`; if no partner expectation exists, use
`EXP_NONE`. A technical failure in one coder does not make the remaining coder
authoritative; the adjudicator independently codes the rationale.

## 9. API accounting schema

For every attempt, retain an accounting row with:

- `blind_id`;
- `role` (`coder1`, `coder2`, `adjudicator`);
- provider, model, temperature, top_p, max_tokens, thinking mode;
- request ID when available;
- attempt number;
- success/failure;
- invalid-output flag;
- technical retry reason;
- timestamp;
- retained-result flag;
- final status.

The API accounting file is `QWEN_EXPECTATION_API_ACCOUNTING.csv`. Credentials,
authorization headers, and raw secret values are never written to this file.

## 10. Frozen execution order

1. Verify the pre-execution check and all frozen hashes.
2. Run exactly one logical coding task per rationale per coder; no content-based
   reruns.
3. Save raw coder outputs and API accounting before adjudication.
4. Compute agreement, kappa, confusion matrix, disagreement count, LOW rate,
   invalid count, and retry count before final adjudication.
5. Adjudicate only the pre-specified queue.
6. Freeze final labels and QC before any expectation-to-strategy or
   expectation-to-contribution analysis.

## 11. Integrity statement

At the time of this amendment, no Qwen expectation category distribution has
been observed. This amendment does not use Qwen contribution, Qwen strategy,
cross-model behavioral results, or any Qwen expectation outcome to alter the
taxonomy or coding rules.
