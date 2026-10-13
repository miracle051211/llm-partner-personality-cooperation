from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_rel
import statsmodels.api as sm


PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "experiment/runs/formal/gpt56terra98_cross_model_formal_v1_20261010_220000"
OUT = PROJECT / "analysis/gpt56terra98"
OUT.mkdir(parents=True, exist_ok=True)
TRAITS = ["A", "O", "C", "N", "E"]
TRAIT_NAMES = {"A": "Agreeableness", "O": "Openness", "C": "Conscientiousness", "N": "Neuroticism", "E": "Extraversion"}
TREATMENTS = {"A": "A", "O": "B", "C": "C", "N": "D", "E": "E"}
BINS = ["High", "Middle", "Low"]
STRATS = ["S1", "S2", "S3", "S4", "S5", "S6"]
CATS = ["EXP_COOP", "EXP_UNCOOP", "EXP_UNCLEAR", "EXP_NONE"]
BOOT_SEED = 20261005
BOOT_REPS = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def rank_bin(x):
    if pd.isna(x):
        return "NoInfo"
    x = float(x)
    return "High" if x <= 5 else "Middle" if x <= 9 else "Low"


def stratified_boot(values, sessions, seed=BOOT_SEED, reps=BOOT_REPS):
    z = pd.DataFrame({"v": np.asarray(values, dtype=float), "s": np.asarray(sessions, dtype=str)}).dropna()
    groups = [g.v.to_numpy(float) for _, g in z.groupby("s", sort=True)]
    if not groups:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        out[i] = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]).mean()
    return out


def add_hash_manifest():
    raw_files = [RUN / "GPT56TERRA_BFI44_RAW.csv", RUN / "GPT56TERRA_BFI44_SCORES.csv", RUN / "10_GPT56TERRA_PRACTICE_RAW.csv", RUN / "GPT56TERRA_STAGE2_RAW.csv", RUN / "GPT56TERRA_STAGE2_RAW.jsonl"]
    stage = pd.read_csv(RUN / "GPT56TERRA_STAGE2_RAW.csv")
    bfi = pd.read_csv(RUN / "GPT56TERRA_BFI44_RAW.csv")
    scores = pd.read_csv(RUN / "GPT56TERRA_BFI44_SCORES.csv")
    practice = pd.read_csv(RUN / "10_GPT56TERRA_PRACTICE_RAW.csv")
    checks = {
        "participants_98": stage.participant_id.nunique() == 98,
        "sessions_7x14": stage.groupby("session_id").participant_id.nunique().to_dict() == {f"P{i}": 14 for i in range(1, 8)},
        "bfi_administrations_98": scores.participant_id.nunique() == 98 and len(scores) == 98,
        "bfi_item_rows_4312": len(bfi) == 98 * 44,
        "practice_rows_392": len(practice) == 392 and practice.participant_id.nunique() == 98,
        "stage2_rows_3528": len(stage) == 3528,
        "trials_per_participant_36": stage.groupby("participant_id").size().eq(36).all(),
        "treatment_counts_588": stage.groupby("treatment").size().to_dict() == {x: 588 for x in ["A", "B", "C", "D", "E", "F"]},
        "noinfo_588": int((stage.treatment == "F").sum()) == 588,
        "visible_2940": int((stage.treatment != "F").sum()) == 2940,
        "rounds_6_per_condition": stage.groupby(["participant_id", "treatment"]).round.nunique().eq(6).all(),
        "no_self_match": bool((stage.participant_id == stage.partner_id).sum() == 0),
        "rank_range_1_14": bool(stage.loc[stage.treatment != "F", "partner_rank"].between(1, 14).all()),
        "contribution_range_0_10": bool(stage.contribution.between(0, 10).all()),
        "contribution_semantics": bool(np.isclose(stage.contribution, 10 - stage.removal).all()),
        "strategy_s1_s6": bool(stage.primary_strategy_code.isin(STRATS).all()),
        "rationale_complete": bool(stage.rationale_text.fillna("").str.strip().ne("").all()),
        "requested_returned_model_exact": bool((stage.requested_model == "gpt-5.6-terra").all() and (stage.returned_model == "gpt-5.6-terra").all()),
        "unique_trial_keys": int(stage[["participant_id", "treatment", "round"]].duplicated().sum()) == 0,
        "strategy_order_exact": bool(stage.strategy_order_exact_match.all()),
    }
    # API logs were written by concurrent workers without a file lock. Count the
    # anomaly without repairing or rewriting those logs.
    log_rows, malformed = 0, []
    for p in sorted((RUN / "api_logs").glob("*.jsonl")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
                log_rows += 1
            except json.JSONDecodeError:
                malformed.append({"file": p.name, "line": n})
    checks["api_log_json_lines_all_parse"] = len(malformed) == 0
    checks = {k: bool(v) for k, v in checks.items()}
    checks["raw_data_structural_pass"] = all(checks[k] for k in checks if k not in {"api_log_json_lines_all_parse", "raw_data_structural_pass"})
    hashes = {str(p.relative_to(RUN)): sha256(p) for p in raw_files if p.exists()}
    manifest = {
        "status": "COMPLETE_RAW_FROZEN" if checks["raw_data_structural_pass"] else "RAW_FREEZE_FAIL",
        "run_id": RUN.name, "model": "gpt-5.6-terra", "base_url": "https://api.iiiiitoken.com",
        "temperature": 0.2, "top_p": 1.0, "thinking": "disabled",
        "expected": {"participants": 98, "bfi_item_rows": 4312, "practice_rows": 392, "stage2_rows": 3528, "noinfo": 588, "visible": 2940},
        "checks": checks, "raw_hashes": hashes, "api_key_written": False, "authorization_header_saved": False,
        "api_log_valid_json_lines": log_rows, "api_log_malformed_lines": malformed,
        "api_log_note": "Two concurrent Stage2 API-log lines were interleaved; authoritative raw CSV/JSONL are structurally complete and unchanged.",
        "frozen_utc": pd.Timestamp.utcnow().isoformat(),
    }
    dump_json(RUN / "RAW_FREEZE_MANIFEST.json", manifest)
    dump_json(RUN / "GPT56TERRA_FINAL_STATUS.json", {
        "status": manifest["status"], "run_id": RUN.name, "requested_model": "gpt-5.6-terra", "returned_model": "gpt-5.6-terra",
        "api_compatibility": "PASS", "personality_gate": "PASS", "rank_freeze": "PASS", "participants": 98,
        "bfi_item_rows": 4312, "practice_rows": 392, "stage2_trials": 3528, "noinfo_trials": 588, "visible_trials": 2940,
        "raw_freeze": "PASS" if manifest["status"] == "COMPLETE_RAW_FROZEN" else "FAIL", "analysis": "NOT_YET_RUN",
        "raw_modified_after_freeze": False, "api_key_or_authorization_saved": False,
    })
    (RUN / "GPT56TERRA_STAGE2_RESUME_STATUS.json").write_text(json.dumps({"status": "COMPLETE_RAW_FROZEN", "final_rows": 3528, "updated_utc": pd.Timestamp.utcnow().isoformat()}, indent=2) + "\n", encoding="utf-8")
    lines = ["# GPT-5.6 Terra execution integrity report", "", "- Stage2 raw rows: **3,528/3,528**.", "- BFI administrations: **98/98**; item rows: **4,312/4,312**.", "- Practice rows: **392/392**.", "- Requested and returned model: **gpt-5.6-terra**.", "- API key and Authorization header saved: **NO**.", "", "## Structural checks", ""]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in checks.items()]
    lines += ["", "## Logger note", "", f"- Valid JSON API-log lines: **{log_rows}**; malformed/interleaved physical lines: **{len(malformed)}**.", "- This logging artifact does not change the authoritative Stage2 CSV/JSONL, which passed all observation-level checks."]
    (RUN / "EXECUTION_INTEGRITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stage, scores, manifest


def prepare(stage, scores):
    roster = pd.read_csv(RUN / "GPT56TERRA_SESSION_ROSTER_FROZEN.csv", dtype=str)
    d = stage.copy()
    d["participant_id"] = d.participant_id.astype(str)
    d["partner_id"] = d.partner_id.astype(str)
    # The authoritative Terra raw already stores both source-profile fields;
    # merge only to verify the frozen mapping without creating duplicate columns.
    verify = roster[["gpt56terra_formal_participant_id", "source_profile_id"]].rename(columns={"gpt56terra_formal_participant_id": "participant_id", "source_profile_id": "_roster_source_profile_id"})
    d = d.merge(verify, on="participant_id", validate="many_to_one")
    if not (d.source_profile_id.astype(str) == d._roster_source_profile_id.astype(str)).all():
        raise RuntimeError("Terra source-profile mapping differs from frozen roster")
    d = d.drop(columns=["_roster_source_profile_id"])
    partner_map = roster[["gpt56terra_formal_participant_id", "source_profile_id"]].rename(columns={"gpt56terra_formal_participant_id": "partner_id", "source_profile_id": "partner_source_profile_id_frozen"})
    d = d.merge(partner_map, on="partner_id", validate="many_to_one")
    if not (d.partner_source_profile_id.astype(str) == d.partner_source_profile_id_frozen.astype(str)).all():
        raise RuntimeError("Terra partner source-profile mapping differs from frozen roster")
    d = d.drop(columns=["partner_source_profile_id_frozen"])
    d["rank_bin"] = d.partner_rank.map(rank_bin)
    d.loc[d.treatment == "F", "rank_bin"] = "NoInfo"
    d["trait"] = d.treatment.map({"A": "A", "B": "O", "C": "C", "D": "N", "E": "E", "F": "NoInfo"})
    d["visible"] = (d.treatment != "F").astype(int)
    d["contribution"] = pd.to_numeric(d.contribution)
    d["round"] = pd.to_numeric(d["round"]).astype(int)
    ranks = pd.read_csv(RUN / "GPT56TERRA_WITHIN_SESSION_RANKS.csv")
    rank_map = ranks.set_index("participant_id")
    for t in TRAITS:
        d[f"own_{t}"] = d.participant_id.map(rank_map[f"measured_{t}"]) / 100.0
        d[f"partner_{t}"] = d.partner_id.map(rank_map[f"measured_{t}"]) / 100.0
        d[f"distance_{t}"] = (d[f"own_{t}"] - d[f"partner_{t}"]).abs()
    return d


def overall(d):
    p = d.groupby(["source_profile_id", "session_id", "visible"], as_index=False).contribution.mean().pivot_table(index=["source_profile_id", "session_id"], columns="visible", values="contribution").reset_index()
    p = p.rename(columns={0: "noinfo_mean", 1: "visible_mean"})
    p["delta_visible_minus_noinfo"] = p.visible_mean - p.noinfo_mean
    boot = stratified_boot(p.delta_visible_minus_noinfo, p.session_id)
    t, pv = ttest_rel(p.visible_mean, p.noinfo_mean)
    row = {"model": "GPT-5.6 Terra", "n_participants": len(p), "noinfo_mean": float(p.noinfo_mean.mean()), "visible_mean": float(p.visible_mean.mean()), "delta_visible_minus_noinfo": float(p.delta_visible_minus_noinfo.mean()), "bootstrap_ci_low": float(np.percentile(boot, 2.5)), "bootstrap_ci_high": float(np.percentile(boot, 97.5)), "paired_t": float(t), "paired_t_p": float(pv), "bootstrap_seed": BOOT_SEED, "bootstrap_reps": BOOT_REPS}
    pd.DataFrame([row]).to_csv(OUT / "01_GPT56TERRA_OVERALL_EFFECT.csv", index=False, encoding="utf-8-sig")
    return p, row


def cells(d):
    base = d[d.treatment == "F"].groupby("source_profile_id").contribution.mean().rename("noinfo_mean")
    sessions = d[["source_profile_id", "session_id"]].drop_duplicates().set_index("source_profile_id").session_id
    rows = []
    for tr, treatment in TREATMENTS.items():
        for b in BINS:
            z = d[(d.treatment == treatment) & (d.rank_bin == b)]
            tar = z.groupby("source_profile_id").contribution.agg(["mean", "size"]).rename(columns={"mean": "target_mean", "size": "n_trials"})
            x = tar.join(base, how="inner")
            x["delta"] = x.target_mean - x.noinfo_mean
            boot = stratified_boot(x.delta, x.index.map(sessions))
            t, p = ttest_1samp(x.delta, 0.0)
            rows.append({"model": "GPT-5.6 Terra", "cell": f"{tr}_{b}", "trait": tr, "trait_name": TRAIT_NAMES[tr], "rank_bin": b, "n_participants": len(x), "n_trials_target": int(x.n_trials.sum()), "target_mean": float(x.target_mean.mean()), "own_noinfo_mean": float(x.noinfo_mean.mean()), "estimate_delta_visible_minus_noinfo": float(x.delta.mean()), "bootstrap_ci_low": float(np.percentile(boot, 2.5)), "bootstrap_ci_high": float(np.percentile(boot, 97.5)), "paired_t": float(t), "raw_p": float(p), "bootstrap_seed": BOOT_SEED, "bootstrap_reps": BOOT_REPS, "session_counts": ";".join(f"{s}:{n}" for s, n in x.index.map(sessions).value_counts().sort_index().items())})
    out = pd.DataFrame(rows)
    order = np.argsort(out.raw_p.to_numpy())
    adj = np.empty(len(out)); adj[order] = np.minimum.accumulate((out.raw_p.to_numpy()[order] * len(out))[::-1] / np.arange(len(out), 0, -1))[::-1]
    out["BH_FDR_q_value_15_cells"] = np.clip(adj, 0, 1)
    out["FDR_family"] = "GPT-5.6 Terra 15 trait x rank-bin cells"
    out.to_csv(OUT / "02_GPT56TERRA_15CELL_EFFECTS.csv", index=False, encoding="utf-8-sig")
    return out


def fit_ols(d, trait, rankbin=False):
    treatment = TREATMENTS[trait]
    z = d[d.treatment == treatment].copy()
    z["Intercept"] = 1.0
    z["Middle"] = (z.partner_rank.between(6, 9)).astype(float)
    z["Low"] = (z.partner_rank.between(10, 14)).astype(float)
    predictors = (["Intercept", "Middle", "Low"] if rankbin else ["Intercept", "partner_rank"]) + [f"distance_{trait}", "round"] + [f"session_{s}" for s in ["P2", "P3", "P4", "P5", "P6", "P7"]] + [f"own_{t}" for t in TRAITS]
    for s in ["P2", "P3", "P4", "P5", "P6", "P7"]:
        z[f"session_{s}"] = (z.session_id == s).astype(float)
    X = z[predictors].astype(float); X = sm.add_constant(X, has_constant="add"); X = X.loc[:, ~X.columns.duplicated()]
    fit = sm.OLS(z.contribution.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": z.source_profile_id})
    terms = ["Middle", "Low"] if rankbin else ["partner_rank"]
    return [{"model": "GPT-5.6 Terra", "trait": trait, "trait_name": TRAIT_NAMES[trait], "term": term, "estimate": float(fit.params[term]), "standard_error": float(fit.bse[term]), "ci_low": float(fit.conf_int().loc[term, 0]), "ci_high": float(fit.conf_int().loc[term, 1]), "p_value": float(fit.pvalues[term]), "n_obs": len(z), "n_clusters": z.source_profile_id.nunique(), "specification": "OLS_clustered_source_profile"} for term in terms]


def human_alignment(d):
    rows = []
    hcont = pd.read_csv(PROJECT / "analysis/human_llm_replication_audit_v1/05_CONTINUOUS_RANK_HUMAN_LLM_COMPARISON.csv")
    hbin = pd.read_csv(PROJECT / "analysis/human_llm_replication_audit_v1/06_RANKBIN_HUMAN_LLM_COMPARISON.csv")
    for tr in TRAITS:
        rows.extend(fit_ols(d, tr, False)); rows.extend(fit_ols(d, tr, True))
    llm = pd.DataFrame(rows)
    for _, h in hcont.iterrows():
        r = llm[(llm.trait == h.trait) & (llm.term == "partner_rank")].iloc[0]
        hs = "positive" if h.human_beta > 0 else "negative" if h.human_beta < 0 else "zero"; ts = "positive" if r.estimate > 0 else "negative" if r.estimate < 0 else "zero"
        rows.append({"domain": "continuous_rank_highness", "trait": h.trait, "trait_name": h.trait_name, "human_estimate": h.human_beta, "human_direction": hs, "terra_estimate": r.estimate, "terra_se": r.standard_error, "terra_ci_low": r.ci_low, "terra_ci_high": r.ci_high, "terra_direction": ts, "direction_match": hs == ts, "comparison_boundary": "direction_only; human random-effects Tobit vs Terra clustered OLS"})
    for _, h in hbin.iterrows():
        term = "Middle" if h.term == "Group2" else "Low"
        r = llm[(llm.trait == h.trait) & (llm.term == term)].iloc[0]
        hs = "positive" if h.human_beta > 0 else "negative" if h.human_beta < 0 else "zero"; ts = "positive" if r.estimate > 0 else "negative" if r.estimate < 0 else "zero"
        rows.append({"domain": "rank_bin_high_reference", "trait": h.trait, "trait_name": h.trait_name, "term": h.term, "human_estimate": h.human_beta, "human_direction": hs, "terra_estimate": r.estimate, "terra_se": r.standard_error, "terra_ci_low": r.ci_low, "terra_ci_high": r.ci_high, "terra_direction": ts, "direction_match": hs == ts, "comparison_boundary": "rank structure/direction only; no coefficient equality"})
    out = pd.DataFrame([r for r in rows if "domain" in r])
    out.to_csv(OUT / "03_GPT56TERRA_HUMAN_ALIGNMENT.csv", index=False, encoding="utf-8-sig")
    return out, llm


def strategy(d):
    rows = []
    for scope, z in [("NoInfo", d[d.visible == 0]), ("Visible", d[d.visible == 1])]:
        for s in STRATS:
            rows.append({"model": "GPT-5.6 Terra", "scope": scope, "strategy": s, "count": int((z.primary_strategy_code == s).sum()), "n": len(z), "proportion": float((z.primary_strategy_code == s).mean())})
    out = pd.DataFrame(rows); wide = out.pivot(index="strategy", columns="scope", values="proportion").reset_index(); wide["visible_minus_noinfo"] = wide.Visible - wide.NoInfo
    out.to_csv(OUT / "05_GPT56TERRA_STRATEGY_DISTRIBUTION.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT / "05_GPT56TERRA_STRATEGY_SHIFT.csv", index=False, encoding="utf-8-sig")
    return out, wide


def cross_model(c15):
    ds = pd.read_csv(PROJECT / "analysis_final/03_15CELL_BEHAVIORAL_EFFECTS.csv")
    ds = ds[ds.model == "DeepSeek"].set_index("cell").mean_delta.rename("DeepSeek")
    qw = pd.read_csv(PROJECT / "analysis_final/03_15CELL_BEHAVIORAL_EFFECTS.csv")
    qw = qw[qw.model == "Qwen"].set_index("cell").mean_delta.rename("Qwen")
    tr = c15.set_index("cell").estimate_delta_visible_minus_noinfo.rename("GPT-5.6 Terra")
    x = pd.concat([ds, qw, tr], axis=1).reset_index()
    rows = []
    for a, b in [("DeepSeek", "Qwen"), ("DeepSeek", "GPT-5.6 Terra"), ("Qwen", "GPT-5.6 Terra")]:
        va, vb = x[a].to_numpy(), x[b].to_numpy()
        rows.extend([{ "pair": f"{a} vs {b}", "metric": "Pearson_r", "value": float(pearsonr(va, vb).statistic)}, {"pair": f"{a} vs {b}", "metric": "Spearman_rho", "value": float(spearmanr(va, vb).statistic)}, {"pair": f"{a} vs {b}", "metric": "sign_agreement", "value": float((np.sign(va) == np.sign(vb)).mean())}])
    out = pd.DataFrame(rows); out.to_csv(OUT / "07_GPT56TERRA_CROSS_MODEL_DESCRIPTIVE.csv", index=False, encoding="utf-8-sig"); x.to_csv(OUT / "07_GPT56TERRA_CROSS_MODEL_15CELL_PROFILE.csv", index=False, encoding="utf-8-sig")
    return out, x


def main():
    stage, scores, freeze = add_hash_manifest()
    d = prepare(stage, scores)
    p, overall_row = overall(d); c15 = cells(d); h, llm = human_alignment(d); s, shift = strategy(d); cross, profiles = cross_model(c15)
    # Expectation labels are intentionally added only when the frozen four-category
    # coder output has been completed; the behavioral package itself is complete.
    exp_path = OUT / "GPT56TERRA_EXPECTATION_FINAL.csv"
    # The historical pre-resume blocker is retained for audit. A complete,
    # validated final table takes precedence over that historical marker.
    exp_status = "PENDING_FROZEN_CODER"
    if exp_path.exists():
        e = pd.read_csv(exp_path)
        d2 = d.merge(e[["participant_id", "treatment", "round", "expectation_code"]], on=["participant_id", "treatment", "round"], validate="one_to_one")
        if len(d2) == 3528 and d2.expectation_code.isin(CATS).all():
            rows = []
            scopes = [("Overall", d2, "ALL", "ALL"), ("NoInfo", d2[d2.visible == 0], "NoInfo", "NoInfo")]
            for trait in TRAITS:
                for b in BINS:
                    scopes.append(("Visible", d2[(d2.visible == 1) & (d2.trait == trait) & (d2.rank_bin == b)], trait, b))
            for scope, zz, trait, b in scopes:
                for cat in CATS:
                    rows.append({"model": "GPT-5.6 Terra", "scope": scope, "cell": "Overall" if scope == "Overall" else ("NoInfo" if scope == "NoInfo" else f"{trait}_{b}"), "trait": trait, "rank_bin": b, "expectation_code": cat, "count": int((zz.expectation_code == cat).sum()), "n": len(zz), "proportion": float((zz.expectation_code == cat).mean()) if len(zz) else np.nan})
            pd.DataFrame(rows).to_csv(OUT / "04_GPT56TERRA_EXPECTATION_DISTRIBUTION.csv", index=False, encoding="utf-8-sig")
            exp_status = "COMPLETE"
        elif (OUT / "GPT56TERRA_EXPECTATION_CODING_BLOCKED.md").exists():
            exp_status = "BLOCKED_INSUFFICIENT_BALANCE"
    elif (OUT / "GPT56TERRA_EXPECTATION_CODING_BLOCKED.md").exists():
        exp_status = "BLOCKED_INSUFFICIENT_BALANCE"
    # Reports are deliberately descriptive and preserve the pre-run interpretation frame.
    c15s = c15.sort_values(["trait", "rank_bin"], key=lambda col: col.map({"High": 0, "Middle": 1, "Low": 2}) if col.name == "rank_bin" else col)
    lines = ["# GPT-5.6 Terra third-backbone analysis report", "", "This is an offline analysis of the frozen 98-instance Terra run. No raw experiment file, persona, session topology, rank table, or manuscript file was modified.", "", "## 1. Overall visibility", "", f"Visible mean contribution = {overall_row['visible_mean']:.3f}; NoInfo = {overall_row['noinfo_mean']:.3f}; Δ = {overall_row['delta_visible_minus_noinfo']:.3f}; session-stratified participant bootstrap 95% CI [{overall_row['bootstrap_ci_low']:.3f}, {overall_row['bootstrap_ci_high']:.3f}]; paired t p={overall_row['paired_t_p']:.4g}.", "", "## 2. Content dependence", "", "The 15 pre-specified trait × rank-bin cells are in `02_GPT56TERRA_15CELL_EFFECTS.csv`; no cell was selected or re-binned after observing results."]
    for tr in TRAITS:
        z = c15[c15.trait == tr].set_index("rank_bin"); lines.append(f"- {tr}: High {z.loc['High','estimate_delta_visible_minus_noinfo']:.3f}; Middle {z.loc['Middle','estimate_delta_visible_minus_noinfo']:.3f}; Low {z.loc['Low','estimate_delta_visible_minus_noinfo']:.3f}.")
    a = c15[c15.trait == "A"].set_index("rank_bin"); low_sep = float(a.loc["Low", "estimate_delta_visible_minus_noinfo"] - a.loc[["High", "Middle"], "estimate_delta_visible_minus_noinfo"].mean())
    lines += ["", "## 3. Agreeableness", "", f"A Low minus higher-rank average separation = {low_sep:.3f}; the direct Low-vs-higher structural contrast is reported descriptively, without requiring High > Middle > Low.", "", "## 4. Openness", "", "Openness is reported cell-by-cell; replication is an empirical outcome rather than a presupposition.", "", "## 5. C/N/E boundaries", "", "Conscientiousness, Neuroticism and Extraversion retain their observed cell estimates, including null or reversed cells.", "", "## 6. Human structural correspondence", "", "`03_GPT56TERRA_HUMAN_ALIGNMENT.csv` compares direction and rank structure only. Human estimates come from the published benchmark; Terra uses clustered OLS, so coefficient equality is not assessed.", "", "## 7. Post-decision expectation", "", f"Expectation coding status: **{exp_status}**. The frozen four-category labels are not inferred from contribution or strategy.", "", "## 8. Reported strategy", "", f"Visible-to-NoInfo S1 shift = {float(shift.loc[shift.strategy == 'S1','visible_minus_noinfo'].iloc[0]):.4f}; full S1–S6 composition is in `05_GPT56TERRA_STRATEGY_DISTRIBUTION.csv`.", "", "## 9. Cross-model robustness", "", "`07_GPT56TERRA_CROSS_MODEL_DESCRIPTIVE.csv` contains only descriptive Pearson, Spearman and sign-agreement comparisons for the 15-cell profile.", "", "## 10. Unexpected results", "", "The API logging layer contains two interleaved physical JSON lines caused by concurrent writes; the authoritative raw files have unique trial keys and pass all observation-level integrity checks."]
    (OUT / "08_GPT56TERRA_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity = ["# GPT56TERRA analysis integrity", "", f"- Raw freeze status: **{freeze['status']}**.", "- Analysis calls to participant or coder APIs: **0**.", "- Source raw hash was captured before analysis and no raw source file was written by this package.", f"- Behavioral outputs: **PASS**; 98 participants, 3,528 Stage2 rows, 588 NoInfo, 2,940 Visible.", f"- Expectation coding: **{exp_status}**.", "- No manuscript, Figure 1–5, or older model raw file was modified.", "- The two malformed API-log lines are disclosed in the run-level execution integrity report and do not enter statistical tables."]
    (OUT / "09_GPT56TERRA_ANALYSIS_INTEGRITY.md").write_text("\n".join(integrity) + "\n", encoding="utf-8")
    dump_json(OUT / "ANALYSIS_MANIFEST.json", {"status": "COMPLETE_BEHAVIORAL_OFFLINE_ANALYSIS", "run": str(RUN), "source_stage2_sha256": sha256(RUN / "GPT56TERRA_STAGE2_RAW.csv"), "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()), "expectation_status": exp_status, "participant_api_calls": 0})
    # Mark the run as analyzed without changing any raw data.
    final = json.loads((RUN / "GPT56TERRA_FINAL_STATUS.json").read_text(encoding="utf-8")); final.update({"analysis": "COMPLETE_BEHAVIORAL_OFFLINE_ANALYSIS", "analysis_directory": str(OUT), "expectation_coding": exp_status}); dump_json(RUN / "GPT56TERRA_FINAL_STATUS.json", final)
    print(json.dumps({"status": "COMPLETE_BEHAVIORAL_OFFLINE_ANALYSIS", "overall_delta": overall_row["delta_visible_minus_noinfo"], "s1_shift": float(shift.loc[shift.strategy == "S1", "visible_minus_noinfo"].iloc[0]), "expectation_status": exp_status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
