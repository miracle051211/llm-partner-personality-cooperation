# Local pre-submission data and code package

This package supports the current MGS manuscript, *Partner personality cues and cooperation in persona-conditioned language-model agents*. It contains the formal three-implementation study only: 98 persona-conditioned instances per implementation, seven fixed groups of 14, 36 formal trials per instance and 3,528 formal trials per implementation (10,584 total). Recorded implementation labels are DeepSeek-V4.1-Flash, Qwen-3.8-Flash and GPT-5.6-Terra. Interface-reported labels do not independently establish model weights or backend routing.

This is a local pre-submission package. No repository upload, publication, persistent identifier, DOI, or public-availability claim is made. Licensing and redistribution decisions remain to be completed before any public release, including rights for reference-profile derivatives, BFI item wording and the adapted lexical resource. No author, affiliation, title-page or cover-letter information is included.

## Layout and source authority

Copied assets preserve their workspace-relative paths. `PROVENANCE_MANIFEST.json` records the relative source and package path, SHA-256 and byte size for every copied source, together with any sanitization. `PACKAGE_SHA256.json` separately hashes the final package files other than itself. `SANITIZATION_REPORT.json` records checks and redactions without credential values. `VALIDATION_REPORT.json` records the executed lightweight checks.

| Package path | Contents and role |
|---|---|
| `experiment/runs/formal/formal_study1_98_v1_1_20261002_172630/` | DeepSeek authoritative `11_FORMAL_STAGE2_RAW.csv` and JSONL; 98 participants and shared fixed persona records; locked matching/order; independent BFI score and rank tables |
| `experiment/runs/formal/qwen98_cross_model_formal_v1_20261007_115200/` | Qwen authoritative `QWEN98_STAGE2_RAW.csv` and JSONL; final 4,312 BFI item records and 98 scores/ranks; frozen mechanics |
| `experiment/runs/formal/gpt56terra98_cross_model_formal_v1_20261010_220000/` | Terra authoritative `GPT56TERRA_STAGE2_RAW.csv` and JSONL; frozen roster, 4,312 BFI item records, scores, ranks, and inventory items |
| `experiment/qwen98_cross_model_formal_prereg_v1/` | Cross-model source-profile map, group membership, matching topology, treatment order, actual shared strategy option order, prompt inventory and seeds |
| `formal_study1_98_execution_adapter_v1_1/` | Frozen 98-row source-profile draw and final treatment-order schedule |
| `formal_study1_98_analysis_v1/08_EXPECTATION_FINAL.csv` | DeepSeek final four-category expectation labels, linked to trial identifiers |
| `analysis_final/QWEN_EXPECTATION_FINAL.csv` | Qwen final four-category labels and recorded coding provenance |
| `analysis/gpt56terra98/GPT56TERRA_EXPECTATION_FINAL.csv` | Terra final four-category labels and recorded coding provenance |
| `formal_study1_98_analysis_v1/` | Existing DeepSeek ordinary-bootstrap outputs and S10 report diagnostics; coding rubric and agreement table |
| `analysis_final/` | Final DeepSeek/Qwen stratified overall and 15-cell effects, strategy compositions/S1 shifts, paired cell and profile diagnostics, expectation distribution; frozen Qwen coding rules |
| `PUBLISHED_BENCHMARK_REPRODUCTION/` | Current DeepSeek/Qwen round/group fixed-effect OLS continuous and bin contrasts, rather than the historical Tobit analysis |
| `analysis/gpt56terra98/` | Terra final overall/15-cell effects, original OLS alignment and rank-highness display table, expectation/strategy compositions, coding rules and rank-coordinate audit |
| `experiment/shared_pipeline/` | Inspected Scheme A percentile mapping, within-group ranking, matching and group allocation modules and helpers |
| `experiment_personality_induction/` | Inspected fixed v3_game_safe persona generator, lexical resource, nine-level qualifiers and imported helpers |

Treatment codes A/B/C/D/E/F mean agreeableness/openness/conscientiousness/neuroticism/extraversion/NoInfo. Trait abbreviations A/O/C/N/E are distinct from treatment codes. Numeric partner rank 1 means highest and 14 lowest within the fixed group. Analysis bins are High 1–5, Middle 6–9, Low 10–14. Rank-highness is `7.5 - numeric_rank`. Contributions are `10 - removal`, integers from 0 through 10. Final expectation labels are EXP_COOP, EXP_UNCOOP, EXP_UNCLEAR and EXP_NONE; all four remain in denominators. Strategies S1–S6 are nominal post-decision reports.

The shared 98 persona strings and prompt levels are archived in the DeepSeek frozen JSONL. Duplicate per-instance text directories are omitted. Prompt construction uses `build_prompt_from_levels`; the imported legacy control-score helper is included only because the generator imports it. The study uses Scheme A empirical percentiles, not that legacy helper's control-score route.

## Lightweight offline validation

Use Python 3.10 or later with pandas. The tested local environment was Python with pandas 3.0.1 and numpy 2.3.5; the validator itself uses stdlib and pandas and the stdlib-only fixed persona builder. No installation, network, API call or model inference is performed by the validator.

From this package directory run:

```text
python validate_release.py
```

The command checks the copied-file manifest; CSV/JSONL trial key agreement; 98 instances, seven groups, 36 rounds and six treatments of 588 rows; unique identifiers; integer 0–10 decisions and contribution conversion; six distinct within-group partners; rank permutations and partner-rank links; final coding coverage and category counts; all 45 equally weighted instance cell estimates and overall Visible–NoInfo estimates; Terra's algebraic rank-coordinate conversion; and exact rebuilding of all 98 stored persona strings. It writes `VALIDATION_REPORT.json` and returns nonzero if a check fails. It does not modify source records.

The validator reconstructs descriptive point estimates and frozen coding counts. It does not rerun bootstrap uncertainty, paired tests, regressions, model coders, or figures. Those archived statistical outputs retain their original definitions and precision. Some archived expectation distributions contain auxiliary Wilson interval columns; the current manuscript uses the category counts/proportions as descriptive compositions and does not present those columns as primary inference.

## Archived scripts and dependencies

All copied scripts were inspected before copying. No archived study script was executed during package preparation. Their end-to-end execution in this relocated package is **unverified**. Run historical scripts only in a disposable working copy because they can overwrite archived outputs. `validate_release.py` is the supported package entry point for the checks actually performed.

| Script | Dependencies and execution boundary |
|---|---|
| `analysis_final/run_all.py` | numpy, pandas, scipy, statsmodels. Uses copied DeepSeek/Qwen raw data, participant map, final DeepSeek coding and strategy crosswalk. Runs 5,000-resample analyses and additional historical diagnostics; writes reports and output tables. Not run here. |
| `formal_study1_98_analysis_v1/scripts/run_formal_analysis.py` | numpy, pandas, scipy, statsmodels. Historical ordinary-bootstrap and supplementary analysis. Its full main also reads a coding rubric from an omitted pilot directory, and expects a tables output directory. Archived for method traceability, not an immediately executable default. |
| `formal_study1_98_analysis_v1/scripts/finalize_formal_analysis.py` | numpy, pandas, scipy, statsmodels, matplotlib, seaborn. Historical coding finalization uses omitted individual coder tables/private blind key and writes older figures. Final labels are supplied instead. |
| `analysis/gpt56terra98/run_gpt56terra_analysis.py` | numpy, pandas, scipy, statsmodels. Frozen Terra specification is retained. Full main requires omitted practice records, human audit inputs and API logs and rewrites run manifests/status. It is not the validation entry point. |
| `analysis/gpt56terra98/generate_gpt56terra_expectation_outputs.py` | pandas + stdlib. Uses supplied Terra raw/final labels and writes descriptive distributions; source inspected, execution not run here. |
| `PUBLISHED_BENCHMARK_REPRODUCTION/build_reproduction_package.py` | numpy, pandas, statsmodels, matplotlib, python-docx. Fits current DeepSeek/Qwen OLS and builds historical figures/reports/Word output at import time. Its private root was replaced with a relative root; relocation is unverified. The manuscript's human coefficients/significance markers remain as stated in the manuscript, not the historical script's synthesized human interval/significance labels. |
| Persona generator and mapping/ranking/matching modules | Generator itself uses stdlib. Mapping/ranking/matching use numpy/pandas plus copied local helpers. Frozen configuration and full orchestration are not bundled. The shared matching module supports other blocks historically; this study uses only its formal six-treatment matching records. Group labels in generic allocation code can differ from the adapter's recorded P1–P7 labels; frozen rosters are authoritative. |

The historical Qwen Tobit pipeline and DeepSeek final-correction scripts were inspected and omitted because their default execution includes pilot dependencies and analyses not used for the current manuscript. API execution adapters and coder executors are also omitted. This package supports inspection and offline checks of the recorded study; it is not a complete new-inference runner.

## Sanitization, exclusions and remaining limits

Sanitization ran on the entire allowlist before any file was copied. Checks covered recognizable provider keys, quoted credential assignments, Bearer credentials, private-key headers, private workstation paths and email addresses. No credential pattern required redaction. Private workspace/library paths were removed from four copied files; a benchmark script root was made relative, participant-map and BFI-score path prefixes were made relative, and an external private library PDF path was replaced with an unavailable-reference placeholder. Original sources were not edited. Source/package checksums make these changes explicit. Pattern checks are a targeted local audit, not a proof that every possible secret representation is absent.

Actual `.env` files, API logs, execution checkpoints, caches, pilot observations, unrelated extensions, author details and publication correspondence are excluded. Logs are not required for the validator and were not copied to satisfy historical script dependencies. No heavy analysis or inference was rerun.

Missing or deliberately omitted assets limit full reproduction:

- DeepSeek per-item BFI raw responses are not supplied; its frozen 98 score records are supplied. Qwen and Terra final item tables are supplied.
- The original 463-row reference library and source SAV deposits are not bundled; the sampled 98 profiles, source identifiers/DOIs, percentiles, levels and fixed prompts are supplied. Full reference-population reconstruction and percentile recalculation were not verified here.
- Individual coder pass/adjudication records and private blinding keys are omitted; final trial-linked labels, rules and recorded agreement summaries are supplied. Coder validity and agreement are not independently re-estimated.
- Practice records, complete API settings, complete physical logs and full generation orchestration are omitted. The manuscript records missing DeepSeek base URL and Qwen Stage-2 token settings and the Terra concurrent-log limitation; this package does not fill those gaps.
- Human participant-level data are unavailable. Human comparisons concern the published benchmark; exact human p-values are not invented.
- Complete inferential and figure regeneration remains unverified. Frozen bootstrap, regression and coding outputs are supplied; only the checks in the validation report were executed.
- Public licensing, repository selection and availability statements remain open. Nothing in this local package implies that it is publicly available.
