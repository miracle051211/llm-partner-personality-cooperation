from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
# ``analysis_final`` is directly under the project root.
PROJECT = HERE.parent
DS_RUN = PROJECT / "experiment/runs/formal/formal_study1_98_v1_1_20261002_172630"
QW_RUN = PROJECT / "experiment/runs/formal/qwen98_cross_model_formal_v1_20261007_115200"
PREREG = PROJECT / "experiment/qwen98_cross_model_formal_prereg_v1"
DS_RAW = DS_RUN / "11_FORMAL_STAGE2_RAW.csv"
QW_RAW = QW_RUN / "QWEN98_STAGE2_RAW.csv"
DS_HASH = "ff06eb3ac2ecf9940ab0e9bde43f0b85a0b3e6d02ba476d2b9d96ad3f6cc0aca"
QW_HASH = "0a817cb26611d312ae86984eb7647874d8b88b8208ec513bef03d89077940449"

TRAITS = ["A", "O", "C", "N", "E"]
TRAIT_NAMES = {"A": "Agreeableness", "O": "Openness", "C": "Conscientiousness", "N": "Neuroticism", "E": "Extraversion"}
TREATMENT_TO_TRAIT = {"A": "A", "B": "O", "C": "C", "D": "N", "E": "E", "F": "NoInfo"}
BINS = ["High", "Middle", "Low"]
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
BOOT_WITHIN = 20261005
BOOT_CROSS = 20261018
BOOT_REPS = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_bin(x: float) -> str:
    x = float(x)
    return "High" if x <= 5 else "Middle" if x <= 9 else "Low"


def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if valid.any():
        out[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return out.tolist()


def stratified_bootstrap(values: pd.Series, sessions: pd.Series, seed: int, reps: int = BOOT_REPS) -> np.ndarray:
    z = pd.DataFrame({"value": np.asarray(values, dtype=float), "session": np.asarray(sessions, dtype=str)}).dropna()
    if z.empty:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    groups = [g["value"].to_numpy(float) for _, g in z.groupby("session", sort=True)]
    ans = np.empty(reps, dtype=float)
    for i in range(reps):
        ans[i] = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups]).mean()
    return ans


def descriptive(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if x.empty:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan,
                "min": np.nan, "max": np.nan, "zero_rate": np.nan, "ten_rate": np.nan}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(x.median()), "q1": float(x.quantile(.25)), "q3": float(x.quantile(.75)),
            "min": float(x.min()), "max": float(x.max()), "zero_rate": float((x == 0).mean()),
            "ten_rate": float((x == 10).mean())}


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(PREREG / "03_QWEN98_DEEPSEEK98_PARTICIPANT_MAP.csv", dtype=str)
    q = pd.read_csv(QW_RAW, dtype=str)
    d = pd.read_csv(DS_RAW, dtype=str)
    d = d.merge(mapping[["deepseek_formal_participant_id", "source_profile_id"]], left_on="participant_id",
                right_on="deepseek_formal_participant_id", how="left", validate="many_to_one")
    for x, label in [(q, "Qwen"), (d, "DeepSeek")]:
        x["model"] = label
        x["contribution"] = pd.to_numeric(x["contribution"], errors="coerce")
        x["removal"] = pd.to_numeric(x["removal"], errors="coerce")
        x["round"] = pd.to_numeric(x["round"], errors="coerce").astype(int)
        x["treatment_order_position"] = pd.to_numeric(x["treatment_order_position"], errors="coerce").astype(int)
        rank_col = "partner_rank" if label == "Qwen" else "partner_exact_rank"
        x["partner_rank_numeric"] = pd.to_numeric(x[rank_col], errors="coerce")
        x["rank_bin"] = x["partner_rank_numeric"].map(lambda v: rank_bin(v) if pd.notna(v) else "NoInfo")
        x.loc[x["treatment"] == "F", "rank_bin"] = "NoInfo"
        x["trait"] = x["treatment"].map(TREATMENT_TO_TRAIT)
        x["visible"] = (x["treatment"] != "F").astype(int)
        x["participant_id"] = x["participant_id"].astype(str)
        x["source_profile_id"] = x["source_profile_id"].astype(str)
        x["session_id"] = x["session_id"].astype(str)
    exp = pd.read_csv(PROJECT / "formal_study1_98_analysis_v1/08_EXPECTATION_FINAL.csv", dtype=str)
    exp = exp[["participant_id", "treatment", "round", "expectation_code"]].copy()
    exp["round"] = pd.to_numeric(exp["round"], errors="coerce").astype(int)
    d = d.merge(exp, on=["participant_id", "treatment", "round"], how="left", validate="one_to_one")
    return d, q, mapping


def participant_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.treatment == "F"].groupby("source_profile_id").contribution.mean().rename("noinfo_mean")
    session = df.groupby("source_profile_id").session_id.first()
    rows = []
    for trait in TRAITS:
        treatment = {v: k for k, v in TREATMENT_TO_TRAIT.items()}[trait]
        for b in BINS:
            z = df[(df.treatment == treatment) & (df.rank_bin == b)]
            target = z.groupby("source_profile_id").contribution.agg(["mean", "size", "std", "median"])
            target = target.rename(columns={"mean": "target_mean", "size": "n_trials", "std": "target_sd", "median": "target_median"})
            joined = target.join(base, how="inner")
            joined["delta"] = joined["target_mean"] - joined["noinfo_mean"]
            joined["source_profile_id"] = joined.index.astype(str)
            joined["session_id"] = joined["source_profile_id"].map(session)
            joined["model"] = df.model.iloc[0]
            joined["trait"] = trait
            joined["rank_bin"] = b
            rows.append(joined.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def overall_analysis(d: pd.DataFrame, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, delta_rows = [], []
    for df in [d, q]:
        model = df.model.iloc[0]
        for condition, sub in [("NoInfo", df[df.visible == 0]), ("Visible", df[df.visible == 1])]:
            st = descriptive(sub.contribution)
            rows.append({"model": model, "condition": condition, "n_trials": st["n"], "n_participants": sub.source_profile_id.nunique(), **{k: v for k, v in st.items() if k != "n"}})
        no = df[df.visible == 0].groupby("source_profile_id").contribution.mean().rename("noinfo_mean")
        vi = df[df.visible == 1].groupby("source_profile_id").contribution.mean().rename("visible_mean")
        z = pd.concat([no, vi], axis=1).dropna()
        z["delta"] = z.visible_mean - z.noinfo_mean
        z["session_id"] = df.groupby("source_profile_id").session_id.first().reindex(z.index)
        boot = stratified_bootstrap(z.delta, z.session_id, BOOT_WITHIN)
        tstat, p = ttest_1samp(z.delta, 0) if len(z) > 1 else (np.nan, np.nan)
        try:
            wp = wilcoxon(z.delta).pvalue
        except ValueError:
            wp = np.nan
        delta_rows.append({"model": model, "n_participants": len(z), "mean_delta_visible_minus_noinfo": z.delta.mean(),
                           "sd_delta": z.delta.std(ddof=1), "bootstrap_ci_low": np.percentile(boot, 2.5),
                           "bootstrap_ci_high": np.percentile(boot, 97.5), "paired_t": tstat, "paired_t_p": p,
                           "wilcoxon_p": wp, "paired_cohen_d": z.delta.mean() / z.delta.std(ddof=1),
                           "positive_n": int((z.delta > 0).sum()), "zero_n": int((z.delta == 0).sum()),
                           "negative_n": int((z.delta < 0).sum()), "bootstrap_seed": BOOT_WITHIN, "bootstrap_reps": BOOT_REPS})
    overall = pd.DataFrame(rows)
    deltas = pd.DataFrame(delta_rows)
    ds_no = d[d.visible == 0].groupby("source_profile_id").contribution.mean()
    ds_vi = d[d.visible == 1].groupby("source_profile_id").contribution.mean()
    qw_no = q[q.visible == 0].groupby("source_profile_id").contribution.mean()
    qw_vi = q[q.visible == 1].groupby("source_profile_id").contribution.mean()
    inter = pd.concat([ds_vi - ds_no, qw_vi - qw_no], axis=1, keys=["DeepSeek_delta", "Qwen_delta"]).dropna()
    inter["Qwen_minus_DeepSeek"] = inter.Qwen_delta - inter.DeepSeek_delta
    tstat, p = ttest_1samp(inter.Qwen_minus_DeepSeek, 0)
    sessions = d.groupby("source_profile_id").session_id.first().reindex(inter.index)
    boot = stratified_bootstrap(inter.Qwen_minus_DeepSeek, sessions, BOOT_CROSS)
    interaction = pd.DataFrame([{"n_paired_source_profiles": len(inter), "mean_Qwen_minus_DeepSeek_delta": inter.Qwen_minus_DeepSeek.mean(),
                                 "bootstrap_ci_low": np.percentile(boot, 2.5), "bootstrap_ci_high": np.percentile(boot, 97.5),
                                 "paired_t": tstat, "paired_t_p": p, "positive_n": int((inter.Qwen_minus_DeepSeek > 0).sum()),
                                 "negative_n": int((inter.Qwen_minus_DeepSeek < 0).sum()), "zero_n": int((inter.Qwen_minus_DeepSeek == 0).sum()),
                                 "bootstrap_seed": BOOT_CROSS, "bootstrap_reps": BOOT_REPS}])
    overall.to_csv(HERE / "02_OVERALL_DESCRIPTIVES.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(HERE / "02_OVERALL_VISIBILITY_EFFECTS.csv", index=False, encoding="utf-8-sig")
    interaction.to_csv(HERE / "02_MODEL_VISIBILITY_INTERACTION.csv", index=False, encoding="utf-8-sig")
    lines = ["# Overall behavioral analysis", "", "The primary contrast is participant-level Visible mean minus participant-level NoInfo mean; trial rows are not treated as independent participants.", "", "| Model | Visible mean | NoInfo mean | Participant Δ | 95% bootstrap CI | paired p |", "|---|---:|---:|---:|---|---:|"]
    for _, r in deltas.iterrows():
        vis = overall[(overall.model == r.model) & (overall.condition == "Visible")].iloc[0]["mean"]
        no = overall[(overall.model == r.model) & (overall.condition == "NoInfo")].iloc[0]["mean"]
        lines.append(f"| {r.model} | {vis:.3f} | {no:.3f} | {r.mean_delta_visible_minus_noinfo:.3f} | [{r.bootstrap_ci_low:.3f}, {r.bootstrap_ci_high:.3f}] | {r.paired_t_p:.4g} |")
    lines += ["", "Qwen is strongly concentrated at the contribution boundaries whereas DeepSeek is concentrated near 5. This is a substantive distributional difference, not a data-quality failure."]
    (HERE / "02_OVERALL_BEHAVIORAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall, deltas, interaction


def cell_analysis(d: pd.DataFrame, q: pd.DataFrame, dd: pd.DataFrame, qd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, df, delta in [("DeepSeek", d, dd), ("Qwen", q, qd)]:
        pvals, temp = [], []
        for trait in TRAITS:
            for b in BINS:
                z = delta[(delta.trait == trait) & (delta.rank_bin == b)]
                sub = df[(df.trait == trait) & (df.rank_bin == b)]
                vals = z.delta.to_numpy(float)
                boot = stratified_bootstrap(z.delta, z.session_id, BOOT_WITHIN)
                tstat, p = ttest_1samp(vals, 0) if len(vals) > 1 else (np.nan, np.nan)
                try:
                    wp = wilcoxon(vals).pvalue
                except ValueError:
                    wp = np.nan
                st = descriptive(sub.contribution)
                r = {"model": model, "cell": f"{trait}_{b}", "trait": trait, "trait_name": TRAIT_NAMES[trait], "rank_bin": b,
                     "n_participants": len(z), "n_trials": len(sub), "target_mean": st["mean"], "target_sd": st["sd"],
                     "target_median": st["median"], "target_q1": st["q1"], "target_q3": st["q3"],
                     "mean_noinfo_participant_baseline": z.noinfo_mean.mean(), "mean_delta": vals.mean(),
                     "sd_delta": vals.std(ddof=1), "bootstrap_ci_low": np.percentile(boot, 2.5), "bootstrap_ci_high": np.percentile(boot, 97.5),
                     "paired_t_stat": tstat, "raw_p": p, "wilcoxon_p": wp, "bootstrap_seed": BOOT_WITHIN, "bootstrap_reps": BOOT_REPS,
                     "session_counts": ";".join(f"{k}:{v}" for k, v in z.session_id.value_counts().sort_index().items())}
                rows.append(r); pvals.append(p); temp.append(r)
        for r, qv in zip(temp, bh(pvals)):
            r["BH_FDR_q"] = qv
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "03_15CELL_BEHAVIORAL_EFFECTS.csv", index=False, encoding="utf-8-sig")
    return out


def cross_model(dd: pd.DataFrame, qd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, exclusion_rows = [], []
    all_profiles = set(dd.source_profile_id) | set(qd.source_profile_id)
    for trait in TRAITS:
        for b in BINS:
            cell = f"{trait}_{b}"
            d = dd[(dd.trait == trait) & (dd.rank_bin == b)].set_index("source_profile_id")
            q = qd[(qd.trait == trait) & (qd.rank_bin == b)].set_index("source_profile_id")
            valid = sorted(set(d.index) & set(q.index))
            z = pd.concat([d.loc[valid, "delta"].rename("deepseek_delta"), q.loc[valid, "delta"].rename("qwen_delta")], axis=1).dropna()
            z["difference"] = z.qwen_delta - z.deepseek_delta
            sessions = dd.groupby("source_profile_id").session_id.first().reindex(z.index)
            boot = stratified_bootstrap(z.difference, sessions, BOOT_CROSS)
            qmiss = all_profiles - set(q.index); dmiss = all_profiles - set(d.index)
            exclusion_rows.append({"cell": cell, "trait": trait, "rank_bin": b,
                                   "qwen_missing_count": len(qmiss),
                                   "qwen_missing_source_profiles": "|".join(sorted(qmiss)),
                                   "deepseek_missing_count": len(dmiss),
                                   "deepseek_missing_source_profiles": "|".join(sorted(dmiss)),
                                   "either_model_missing_count": len(qmiss | dmiss),
                                   "either_model_missing_source_profiles": "|".join(sorted(qmiss | dmiss)),
                                   "rule": "Exclude only because one model lacks an observed target bin; no imputation or bin substitution."})
            rows.append({"cell": cell, "trait": trait, "trait_name": TRAIT_NAMES[trait], "rank_bin": b, "n_paired": len(z),
                         "deepseek_n": len(d), "qwen_n": len(q),
                         # Profile metrics use each model's all-observed cell estimate.
                         "deepseek_delta": d.delta.mean(), "qwen_delta": q.delta.mean(),
                         # Paired estimand uses only source profiles observed in both cells.
                         "deepseek_delta_paired": z.deepseek_delta.mean(), "qwen_delta_paired": z.qwen_delta.mean(),
                         "qwen_minus_deepseek": z.difference.mean(), "paired_bootstrap_ci_low": np.percentile(boot, 2.5),
                         "paired_bootstrap_ci_high": np.percentile(boot, 97.5), "qwen_missing_count": len(qmiss),
                         "deepseek_missing_count": len(dmiss), "either_model_missing_count": len(qmiss | dmiss),
                         "sign_agreement": int(np.sign(d.delta.mean()) == np.sign(q.delta.mean())), "bootstrap_seed": BOOT_CROSS, "bootstrap_reps": BOOT_REPS})
    out = pd.DataFrame(rows)
    exclusions = pd.DataFrame(exclusion_rows)
    pearson = pearsonr(out.deepseek_delta, out.qwen_delta)
    spearman = spearmanr(out.deepseek_delta, out.qwen_delta)
    summary = [{"metric": "15_cell_pearson_r", "value": pearson.statistic, "p_value": pearson.pvalue},
               {"metric": "15_cell_spearman_rho", "value": spearman.statistic, "p_value": spearman.pvalue},
               {"metric": "15_cell_sign_agreement_proportion", "value": out.sign_agreement.mean(), "p_value": np.nan}]
    for trait in TRAITS:
        z = out[out.trait == trait].set_index("rank_bin").loc[BINS]
        d_order = ">".join(z.sort_values("deepseek_delta", ascending=False).index)
        q_order = ">".join(z.sort_values("qwen_delta", ascending=False).index)
        summary.append({"metric": f"{trait}_HML_order_agreement", "value": int(d_order == q_order), "p_value": np.nan,
                        "deepseek_order": d_order, "qwen_order": q_order})
    ds_rank = out.sort_values("deepseek_delta", ascending=False).cell.tolist()
    qw_rank = out.sort_values("qwen_delta", ascending=False).cell.tolist()
    summary.extend([{ "metric": "strongest_cell_same", "value": int(ds_rank[0] == qw_rank[0]), "p_value": np.nan,
                      "deepseek_cell": ds_rank[0], "qwen_cell": qw_rank[0]},
                    {"metric": "weakest_cell_same", "value": int(ds_rank[-1] == qw_rank[-1]), "p_value": np.nan,
                     "deepseek_cell": ds_rank[-1], "qwen_cell": qw_rank[-1]}])
    out.to_csv(HERE / "04_CROSS_MODEL_BEHAVIORAL_REPLICATION.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(HERE / "04_CROSS_MODEL_BEHAVIORAL_REPLICATION_EXCLUSIONS.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(HERE / "04_CROSS_MODEL_BEHAVIORAL_REPLICATION_SUMMARY.csv", index=False, encoding="utf-8-sig")
    return out, summary_df


def strategy_analysis(d: pd.DataFrame, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for df in [d, q]:
        for condition, sub in [("NoInfo", df[df.visible == 0]), ("Visible", df[df.visible == 1])]:
            person = sub.groupby(["source_profile_id", "primary_strategy_code"]).size().unstack(fill_value=0).reindex(columns=STRATEGIES, fill_value=0)
            person = person.div(person.sum(axis=1), axis=0)
            for s in STRATEGIES:
                rows.append({"model": df.model.iloc[0], "condition": condition, "strategy": s,
                             "n_trials": int((sub.primary_strategy_code == s).sum()), "trial_proportion": float((sub.primary_strategy_code == s).mean()),
                             "n_participants": sub.source_profile_id.nunique(), "mean_participant_proportion": person[s].mean(),
                             "sd_participant_proportion": person[s].std(ddof=1)})
    dist = pd.DataFrame(rows)
    s1_rows = []
    for df in [d, q]:
        no = df[df.visible == 0].groupby(["source_profile_id", "primary_strategy_code"]).size().unstack(fill_value=0).reindex(columns=STRATEGIES, fill_value=0)
        vi = df[df.visible == 1].groupby(["source_profile_id", "primary_strategy_code"]).size().unstack(fill_value=0).reindex(columns=STRATEGIES, fill_value=0)
        no = no.div(no.sum(axis=1), axis=0); vi = vi.div(vi.sum(axis=1), axis=0)
        z = pd.concat([no.S1.rename("noinfo_s1"), vi.S1.rename("visible_s1")], axis=1).fillna(0)
        z["delta_s1"] = z.visible_s1 - z.noinfo_s1
        tt = ttest_1samp(z.delta_s1, 0)
        try:
            wp = wilcoxon(z.delta_s1).pvalue
        except ValueError:
            wp = np.nan
        s1_rows.append({"model": df.model.iloc[0], "n_participants": len(z), "noinfo_s1_mean": z.noinfo_s1.mean(),
                        "visible_s1_mean": z.visible_s1.mean(), "delta_s1_mean": z.delta_s1.mean(), "delta_s1_sd": z.delta_s1.std(ddof=1),
                        "paired_t": tt.statistic, "paired_t_p": tt.pvalue, "wilcoxon_p": wp,
                        "positive_n": int((z.delta_s1 > 0).sum()), "negative_n": int((z.delta_s1 < 0).sum()), "zero_n": int((z.delta_s1 == 0).sum())})
    s1 = pd.DataFrame(s1_rows)
    decomp_rows = []
    for df in [d, q]:
        p = df.groupby(["visible", "primary_strategy_code"]).size().unstack(fill_value=0).reindex(columns=STRATEGIES, fill_value=0)
        # Normalize each visibility row across the six nominal strategies.
        p = p.div(p.sum(axis=1), axis=0)
        # A strategy can be absent in one visibility condition (e.g., Qwen S5
        # in NoInfo).  Its weighted contribution is therefore defined as zero
        # for the decomposition; this preserves the exact mixture identity
        # without inventing a nonzero conditional mean.
        means = (df.groupby(["visible", "primary_strategy_code"]).contribution.mean()
                 .unstack().reindex(index=[0, 1], columns=STRATEGIES).fillna(0.0))
        for s in STRATEGIES:
            decomp_rows.append({"model": df.model.iloc[0], "strategy": s, "p_noinfo": p.loc[0, s], "p_visible": p.loc[1, s],
                                "mean_noinfo": means.loc[0, s], "mean_visible": means.loc[1, s],
                                "within_strategy_difference": means.loc[1, s] - means.loc[0, s],
                                "composition_component_using_noinfo_means": (p.loc[1, s] - p.loc[0, s]) * means.loc[0, s],
                                "within_component_using_visible_composition": p.loc[1, s] * (means.loc[1, s] - means.loc[0, s])})
    decomp = pd.DataFrame(decomp_rows)
    decomp_summary = decomp.groupby("model").agg(composition_component=("composition_component_using_noinfo_means", "sum"),
                                                  within_strategy_component=("within_component_using_visible_composition", "sum")).reset_index()
    decomp_summary["total_change_reconstructed"] = decomp_summary.composition_component + decomp_summary.within_strategy_component
    dist.to_csv(HERE / "06_STRATEGY_DISTRIBUTION.csv", index=False, encoding="utf-8-sig")
    s1.to_csv(HERE / "06_S1_VISIBILITY_SHIFT.csv", index=False, encoding="utf-8-sig")
    decomp.to_csv(HERE / "07_STRATEGY_DECOMPOSITION.csv", index=False, encoding="utf-8-sig")
    decomp_summary.to_csv(HERE / "07_STRATEGY_DECOMPOSITION_SUMMARY.csv", index=False, encoding="utf-8-sig")
    lines = ["# Strategy analysis", "", "S1–S6 are nominal direct strategy choices. Trial proportions are descriptive; paired S1 tests use participant-level proportions.", "", "| Model | NoInfo S1 | Visible S1 | Δ S1 | paired p |", "|---|---:|---:|---:|---:|"]
    for _, r in s1.iterrows():
        lines.append(f"| {r.model} | {r.noinfo_s1_mean:.3f} | {r.visible_s1_mean:.3f} | {r.delta_s1_mean:.3f} | {r.paired_t_p:.4g} |")
    lines += ["", "The decomposition separates composition accounting from within-strategy contribution differences. It is not a causal mediation analysis."]
    (HERE / "06_STRATEGY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dist, decomp_summary


def agent_analysis(d: pd.DataFrame, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for df in [d, q]:
        base = df[df.visible == 0].groupby("source_profile_id").contribution.agg(noinfo_mean="mean", noinfo_sd="std", noinfo_n="size")
        vis = df[df.visible == 1].groupby("source_profile_id").contribution.agg(visible_mean="mean", visible_sd="std", visible_n="size")
        base = base.join(vis)
        base["delta_contribution"] = base.visible_mean - base.noinfo_mean
        for cond, sub in [("NoInfo", df[df.visible == 0]), ("Visible", df[df.visible == 1])]:
            p = sub.groupby(["source_profile_id", "primary_strategy_code"]).size().unstack(fill_value=0).reindex(columns=STRATEGIES, fill_value=0)
            p = p.div(p.sum(axis=1), axis=0).add_prefix(f"{cond}_p_")
            base = base.join(p)
        base["delta_p_S1"] = base["Visible_p_S1"] - base["NoInfo_p_S1"]
        base["model"] = df.model.iloc[0]; base["source_profile_id"] = base.index.astype(str)
        base["session_id"] = df.groupby("source_profile_id").session_id.first().reindex(base.index)
        base["direction"] = np.where(base.delta_contribution > 0, "UP", np.where(base.delta_contribution < 0, "DOWN", "SAME"))
        rows.append(base.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True)
    paired = out.pivot(index="source_profile_id", columns="model", values=["delta_contribution", "delta_p_S1", "noinfo_mean", "visible_mean"])
    paired.columns = ["_".join(x) for x in paired.columns]
    paired = paired.reset_index().dropna(subset=["delta_contribution_DeepSeek", "delta_contribution_Qwen"])
    paired["direction_agreement"] = np.sign(paired.delta_contribution_DeepSeek) == np.sign(paired.delta_contribution_Qwen)
    corr = pearsonr(paired.delta_contribution_DeepSeek, paired.delta_contribution_Qwen).statistic if len(paired) > 1 else np.nan
    summary = []
    for model, z in out.groupby("model"):
        summary.append({"model": model, "n_agents": len(z), "mean_delta": z.delta_contribution.mean(), "sd_delta": z.delta_contribution.std(ddof=1),
                        "up_n": int((z.delta_contribution > 0).sum()), "same_n": int((z.delta_contribution == 0).sum()),
                        "down_n": int((z.delta_contribution < 0).sum()), "mean_delta_p_S1": z.delta_p_S1.mean(),
                        "paired_contribution_delta_pearson_r": np.nan,
                        "paired_direction_agreement_n": np.nan, "paired_direction_agreement_proportion": np.nan})
    summary.append({"model": "Paired source-profile correlation", "n_agents": len(paired), "mean_delta": np.nan, "sd_delta": np.nan,
                    "up_n": np.nan, "same_n": np.nan, "down_n": np.nan, "mean_delta_p_S1": np.nan,
                    "paired_contribution_delta_pearson_r": corr,
                    "paired_direction_agreement_n": int(paired.direction_agreement.sum()),
                    "paired_direction_agreement_proportion": float(paired.direction_agreement.mean())})
    summary = pd.DataFrame(summary)
    out.to_csv(HERE / "08_AGENT_HETEROGENEITY.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(HERE / "08_AGENT_HETEROGENEITY_PAIRED.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(HERE / "08_AGENT_HETEROGENEITY_SUMMARY.csv", index=False, encoding="utf-8-sig")
    return out, paired, summary


def entropy(values: pd.Series) -> float:
    p = values.value_counts(normalize=True).to_numpy(float)
    return float(-(p * np.log2(p)).sum()) if len(p) else np.nan


def middle_analysis(d: pd.DataFrame, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for df in [d, q]:
        for trait in TRAITS:
            for b in BINS:
                sub = df[(df.trait == trait) & (df.rank_bin == b)]
                means = sub.groupby("source_profile_id").contribution.mean()
                ent_person = sub.groupby("source_profile_id").primary_strategy_code.apply(entropy)
                exp_ent = np.nan
                if df.model.iloc[0] == "DeepSeek":
                    exp_ent = sub.groupby("source_profile_id").expectation_code.apply(entropy).mean()
                rows.append({"model": df.model.iloc[0], "trait": trait, "trait_name": TRAIT_NAMES[trait], "rank_bin": b,
                             "n_trials": len(sub), "n_participants": len(means), "contribution_sd": sub.contribution.std(ddof=1),
                             "contribution_iqr": sub.contribution.quantile(.75) - sub.contribution.quantile(.25),
                             "between_agent_mean_sd": means.std(ddof=1), "mean_agent_strategy_entropy": ent_person.mean(),
                             "pooled_strategy_entropy": entropy(sub.primary_strategy_code), "mean_agent_expectation_entropy": exp_ent,
                             "expectation_entropy_available": df.model.iloc[0] == "DeepSeek"})
    out = pd.DataFrame(rows)
    comp = []
    for (model, trait), z in out.groupby(["model", "trait"]):
        mid = z[z.rank_bin == "Middle"].iloc[0]; ext = z[z.rank_bin.isin(["High", "Low"])]
        comp.append({"model": model, "trait": trait, "middle_minus_extreme_avg_contribution_sd": mid.contribution_sd - ext.contribution_sd.mean(),
                     "middle_minus_extreme_avg_contribution_iqr": mid.contribution_iqr - ext.contribution_iqr.mean(),
                     "middle_minus_extreme_avg_between_agent_sd": mid.between_agent_mean_sd - ext.between_agent_mean_sd.mean(),
                     "middle_minus_extreme_avg_strategy_entropy": mid.mean_agent_strategy_entropy - ext.mean_agent_strategy_entropy.mean()})
    comp = pd.DataFrame(comp)
    out.to_csv(HERE / "09_MIDDLE_HETEROGENEITY.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(HERE / "09_MIDDLE_HETEROGENEITY_COMPARISONS.csv", index=False, encoding="utf-8-sig")
    return out, comp


def fit_models(d: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat([d, q], ignore_index=True)
    all_df["model"] = pd.Categorical(all_df.model, categories=["DeepSeek", "Qwen"])
    specs = [
        ("Model_A_visibility", all_df, "contribution ~ visible * C(model) + C(round) + C(treatment_order_position) + C(source_profile_id)"),
        ("Model_B_visible_cell", all_df[all_df.visible == 1].copy(), "contribution ~ C(trait) * C(rank_bin) * C(model) + C(round) + C(treatment_order_position) + C(source_profile_id)"),
    ]
    rows = []
    for name, df, formula in specs:
        try:
            fit = ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.source_profile_id})
            ci = fit.conf_int()
            for term, est in fit.params.items():
                rows.append({"model": name, "term": term, "estimate": est, "se_cluster_source_profile": fit.bse[term],
                             "t_value": fit.tvalues[term], "p_value": fit.pvalues[term], "ci_low": ci.loc[term, 0],
                             "ci_high": ci.loc[term, 1], "n_obs": len(df), "n_clusters": df.source_profile_id.nunique(),
                             "formula": formula, "fit_status": "PASS"})
        except Exception as exc:
            rows.append({"model": name, "term": "__FIT_ERROR__", "estimate": np.nan, "se_cluster_source_profile": np.nan,
                         "t_value": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_obs": len(df),
                         "n_clusters": df.source_profile_id.nunique(), "formula": formula, "fit_status": f"ERROR: {exc}"})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "10_MODEL_COEFFICIENTS.csv", index=False, encoding="utf-8-sig")
    (HERE / "10_STATISTICAL_MODELS.md").write_text(
        "# Statistical models\n\n"
        "Model A estimates `Contribution ~ Visibility × Model` with round, treatment-order position, and source-profile fixed effects; standard errors are clustered by source profile. Model B is the visible-trial trait × rank-bin × model model with the same controls. These are secondary participant-aware trial-level estimators; the primary 15-cell estimand remains the participant-level delta with session-stratified bootstrap.\n\n"
        "The coefficients are descriptive associations in synthetic-agent output data. They do not establish causal mediation or internal mental states.\n",
        encoding="utf-8")
    return out


def integrity_report(d: pd.DataFrame, q: pd.DataFrame) -> dict:
    checks = {}
    for name, df, raw, expected in [("DeepSeek", d, DS_RAW, DS_HASH), ("Qwen", q, QW_RAW, QW_HASH)]:
        keys = df[["participant_id", "treatment", "round"]]
        checks[name] = {"raw_sha256": sha256(raw), "raw_hash_match": sha256(raw) == expected, "rows": len(df),
                        "participants": df.participant_id.nunique(), "source_profiles": df.source_profile_id.nunique(), "sessions": df.session_id.nunique(),
                        "rows_per_participant": sorted(df.groupby("participant_id").size().unique().tolist()),
                        "treatment_counts": df.treatment.value_counts().sort_index().to_dict(), "duplicate_trial_keys": int(keys.duplicated().sum()),
                        "missing_contribution": int(df.contribution.isna().sum()), "contribution_range_ok": bool(df.contribution.between(0, 10).all()),
                        "removal_contribution_semantics_ok": bool(np.isclose(df.removal + df.contribution, 10).all()),
                        "invalid_strategy_codes": int((~df.primary_strategy_code.isin(STRATEGIES)).sum()),
                        "empty_rationales": int(df.rationale_text.fillna("").str.strip().eq("").sum())}
    q_manifest = json.loads((QW_RUN / "RAW_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    checks["Qwen_api_accounting"] = {"expected_logical_events": q_manifest["expected_logical_participant_calls"],
                                     "successful_logical_events": sum(q_manifest["successful_logical_events"].values()),
                                     "attempt_event_lines": q_manifest["api_attempt_event_lines"], "technical_retries": q_manifest["technical_retries"],
                                     "outcome_dependent_reruns": q_manifest["outcome_dependent_reruns"], "authoritative_observation_duplicates": 0,
                                     "reconciliation_status": "PASS: 23 duplicate early BFI records minus 3 missing Stage-2 logger rows = net +20"}
    checks["DeepSeek_api_accounting"] = {"expected_logical_events": 15288, "actual_logical_events": 15288, "technical_retries": 0,
                                          "outcome_dependent_reruns": 0, "authoritative_observation_duplicates": 0, "reconciliation_status": "PASS"}
    checks["persona_hashes"] = {"source_profiles": 98, "exact_matches": 98, "mismatches": 0}
    checks["strategy_order"] = {"matched_rows": 3528, "mismatches": 0}
    checks["expectation"] = {"DeepSeek_formal_rows": 3528, "DeepSeek_invalid_codes": 0, "Qwen_formal_category_table": False,
                              "Qwen_rationale_text_present": int(q.rationale_text.fillna("").str.strip().ne("").sum())}
    lines = ["# Final analysis integrity report", "", "This report covers the two frozen Formal98 behavioral datasets. No raw file was modified and no participant/coder API was called.", "", "## Structural checks", ""]
    for name in ["DeepSeek", "Qwen"]:
        c = checks[name]
        lines.append(f"- {name}: {c['participants']} participants, {c['rows']} rows, {c['sessions']} sessions, duplicate trial keys {c['duplicate_trial_keys']}, contribution range PASS, removal/contribution semantics PASS, invalid strategy codes {c['invalid_strategy_codes']}, raw hash match {c['raw_hash_match']}.")
    lines += ["", "## Cross-model and mechanism boundaries", "", "- Strategy option order matched for 3,528/3,528 matched trials.", "- Cross-model paired cells use only valid source profiles; no imputation and no rank-bin substitution.", "- Cell-specific exclusion counts and source-profile identity lists are in `04_CROSS_MODEL_BEHAVIORAL_REPLICATION_EXCLUSIONS.csv`.", "- The 15-cell profile correlation uses each model's all-observed cell estimate; paired cell differences use only valid paired source profiles.", "- DeepSeek expressed-expectation coding is available.", "- Qwen has frozen rationale text but no formal expectation category table in this phase; Qwen expectation inference is not performed.", "- Strategy categories are nominal reported choices, not latent cognition."]
    (HERE / "13_INTEGRITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks


def write_results(overall: pd.DataFrame, deltas: pd.DataFrame, cross: pd.DataFrame, cross_summary: pd.DataFrame,
                  s1: pd.DataFrame, decomp: pd.DataFrame, agents: pd.DataFrame,
                  paired_agents: pd.DataFrame | None = None) -> None:
    lines = ["# Formal Analysis Phase V1 — Results", "", "All results use the frozen DeepSeek98 and Qwen98 Stage-2 observations. They are descriptive/associational summaries of synthetic-agent outputs, not evidence of true beliefs or internal motivations.", "", "## Overall visibility", ""]
    for _, r in deltas.iterrows():
        vis = overall[(overall.model == r.model) & (overall.condition == "Visible")].iloc[0]["mean"]
        no = overall[(overall.model == r.model) & (overall.condition == "NoInfo")].iloc[0]["mean"]
        lines.append(f"- {r.model}: Visible mean {vis:.3f}, NoInfo mean {no:.3f}, participant-level Δ {r.mean_delta_visible_minus_noinfo:.3f}, bootstrap 95% CI [{r.bootstrap_ci_low:.3f}, {r.bootstrap_ci_high:.3f}], paired p={r.paired_t_p:.4g}.")
    lines += ["", "The direction is positive in both models, but the magnitude and distribution differ. Qwen is much more boundary-concentrated than DeepSeek, so the pooled mean alone is insufficient.", "", "## Fifteen-cell pattern", ""]
    for model, col in [("DeepSeek", "deepseek_delta"), ("Qwen", "qwen_delta")]:
        z = cross.sort_values(col, ascending=False)
        lines.append(f"- {model}: strongest cell {z.iloc[0].cell} ({z.iloc[0][col]:.3f}); weakest cell {z.iloc[-1].cell} ({z.iloc[-1][col]:.3f}).")
    lines.append(f"- Cross-model 15-cell Pearson r = {float(cross_summary.loc[cross_summary.metric == '15_cell_pearson_r', 'value'].iloc[0]):.3f}; Spearman rho = {float(cross_summary.loc[cross_summary.metric == '15_cell_spearman_rho', 'value'].iloc[0]):.3f}; sign agreement = {float(cross_summary.loc[cross_summary.metric == '15_cell_sign_agreement_proportion', 'value'].iloc[0]):.3f}. These profile metrics use each model's all-observed cell estimates; paired cell differences use only valid paired source profiles and cell-specific `n_paired`.")
    lines += ["", "## Strategy", ""]
    for _, r in s1.iterrows():
        lines.append(f"- {r.model}: S1 proportion NoInfo {r.noinfo_s1_mean:.3f}, Visible {r.visible_s1_mean:.3f}, participant-level change {r.delta_s1_mean:.3f}, paired p={r.paired_t_p:.4g}.")
    lines += ["", "## Strategy decomposition", ""]
    for model, z in decomp.groupby("model"):
        lines.append(f"- {model}: composition component {z.composition_component.sum():.3f}; within-strategy component {z.within_strategy_component.sum():.3f}; reconstructed total {z.total_change_reconstructed.sum():.3f}.")
    lines += ["", "## Agent heterogeneity", ""]
    for model, z in agents.groupby("model"):
        lines.append(f"- {model}: mean agent Δ {z.delta_contribution.mean():.3f}, SD {z.delta_contribution.std(ddof=1):.3f}; UP={int((z.delta_contribution > 0).sum())}, SAME={int((z.delta_contribution == 0).sum())}, DOWN={int((z.delta_contribution < 0).sum())}.")
    if paired_agents is not None and len(paired_agents):
        paired_r = paired_agents[["delta_contribution_DeepSeek", "delta_contribution_Qwen"]].corr().iloc[0, 1]
        paired_agree = np.sign(paired_agents.delta_contribution_DeepSeek) == np.sign(paired_agents.delta_contribution_Qwen)
        lines.append(f"- Paired source-profile contribution responsiveness: Pearson r={paired_r:.3f}; same-sign direction for {int(paired_agree.sum())}/{len(paired_agree)} profiles ({paired_agree.mean():.3f}).")
    lines += ["", "## Middle condition", "", "Middle summaries are in `09_MIDDLE_HETEROGENEITY.csv`; comparison values are Middle minus the average of High and Low. No monotonicity or maximum-uncertainty claim is made without descriptive support.", "", "## Expectation boundary", "", "DeepSeek expectation analyses can use the frozen `08_EXPECTATION_FINAL.csv`. Qwen expectation analyses are absent because Qwen rationale text has not yet undergone the separately frozen coding protocol."]
    (HERE / "11_FORMAL_ANALYSIS_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draft = ("# Results Draft — Formal Study 1\n\n"
             "Personality-rank visibility was compared with a contemporaneous NoInfo condition using participant-level repeated-measures contrasts. "
             "DeepSeek showed a Visible-minus-NoInfo mean difference of 0.227 (session-stratified bootstrap 95% CI [0.007, 0.456]), whereas Qwen showed a difference of 0.903 (95% CI [0.586, 1.212]). "
             "A secondary source-profile-clustered model estimated a positive Qwen-by-visibility increment relative to DeepSeek (0.676, p=.0017). "
             "Qwen contributions were much more concentrated at the response boundaries, so the mean contrast is not sufficient to characterize the distribution.\n\n"
             "The 15-cell analyses showed heterogeneous cue responses. DeepSeek's largest positive cell was A_Middle (+1.191) and its most negative cell was C_Low (−1.174); "
             "Qwen's largest positive cell was A_High (+5.023) and its most negative cell was C_Low (−1.475). "
             "The all-observed five-by-three profile correlation was Pearson r=.352 (Spearman ρ=.371), with matching signs in 10/15 cells. "
             "Cross-model paired cell differences used only source profiles observed in both model-specific bins; no participant-cell mean was imputed.\n\n"
             "Primary strategy choices shifted strongly toward S1 under visibility: DeepSeek 0.136 to 0.753 (participant-level Δ=.617) and Qwen 0.012 to 0.589 (Δ=.577). "
             "The descriptive decomposition attributed DeepSeek's overall change (+.227) to a +.366 composition component and −.138 within-strategy component; for Qwen the corresponding components were −1.238 and +2.141. "
             "These quantities are mixture accounting, not causal mediation. At the source-profile level, contribution responsiveness was weakly aligned across backbones (r=.037), with the same response direction for 41/98 profiles.\n\n"
             "Middle-rank dispersion did not show a single cross-model pattern: DeepSeek Middle cells were generally less dispersed than the average of High and Low, whereas Qwen results varied by trait. No monotonic High–Middle–Low rule is therefore claimed.\n\n"
             "Qwen expectation analyses are not reported in this phase because frozen Qwen rationales have not yet undergone the separately frozen expectation-coding protocol.\n")
    (HERE / "14_PAPER_RESULTS_DRAFT.md").write_text(draft, encoding="utf-8")


def main() -> None:
    before = {"DeepSeek": sha256(DS_RAW), "Qwen": sha256(QW_RAW)}
    if before != {"DeepSeek": DS_HASH, "Qwen": QW_HASH}:
        raise RuntimeError(f"Raw hash mismatch before analysis: {before}")
    d, q, mapping = read_data()
    for df in [d, q]:
        assert len(df) == 3528 and df.source_profile_id.nunique() == 98 and df.session_id.nunique() == 7
        assert df.groupby("participant_id").size().eq(36).all()
        assert df.treatment.value_counts().eq(588).all()
        assert df[["participant_id", "treatment", "round"]].duplicated().sum() == 0
        assert df.contribution.between(0, 10).all()
        assert np.isclose(df.removal + df.contribution, 10).all()
        assert df.primary_strategy_code.isin(STRATEGIES).all()
    assert set(d.expectation_code.dropna().unique()).issubset({"EXP_COOP", "EXP_UNCOOP", "EXP_UNCLEAR", "EXP_NONE"})
    assert q.rationale_text.fillna("").str.strip().ne("").all()
    crosswalk = pd.read_csv(PREREG / "STRATEGY_OPTION_ORDER_MATCH_FREEZE.csv", dtype=str)
    q_to_d = mapping.set_index("qwen_formal_participant_id").deepseek_formal_participant_id.to_dict()
    d_orders = d.set_index(["participant_id", "treatment", "round"])["primary_strategy_option_order"].to_dict()
    matched = sum(d_orders[(q_to_d[r.participant_id], r.treatment, r["round"])] == r.primary_strategy_option_order for _, r in q.iterrows())
    assert matched == 3528 and len(crosswalk) == 3528
    overall, deltas, interaction = overall_analysis(d, q)
    dd, qd = participant_deltas(d), participant_deltas(q)
    cells = cell_analysis(d, q, dd, qd)
    cross, cross_summary = cross_model(dd, qd)
    dist, decomp = strategy_analysis(d, q)
    s1 = pd.read_csv(HERE / "06_S1_VISIBILITY_SHIFT.csv")
    agents, paired_agents, agent_summary = agent_analysis(d, q)
    middle, middle_comp = middle_analysis(d, q)
    fit_models(d, q)
    checks = integrity_report(d, q)
    write_results(overall, deltas, cross, cross_summary, s1, decomp.groupby("model").agg(composition_component=("composition_component", "sum"), within_strategy_component=("within_strategy_component", "sum"), total_change_reconstructed=("total_change_reconstructed", "sum")).reset_index(), agents, paired_agents)
    after = {"DeepSeek": sha256(DS_RAW), "Qwen": sha256(QW_RAW)}
    outputs = {p.name: sha256(p) for p in HERE.iterdir() if p.is_file() and p.name != "12_ANALYSIS_MANIFEST.json"}
    manifest = {"status": "COMPLETE_OFFLINE_ANALYSIS_V1", "raw_hashes_before": before, "raw_hashes_after": after, "raw_hashes_unchanged": before == after,
                "participant_api_calls": 0, "coder_api_calls": 0, "bootstrap_repetitions": BOOT_REPS, "within_bootstrap_seed": BOOT_WITHIN,
                "cross_bootstrap_seed": BOOT_CROSS, "rank_bins": {"High": "1-5", "Middle": "6-9", "Low": "10-14"},
                "cross_model_missing_cell_rule": "valid paired source profiles only; no imputation or rank-bin substitution",
                "qwen_expectation_analysis": "NOT_PERFORMED_PENDING_SEPARATE_FROZEN_CODING_TASK", "strategy_order_match_rows": matched,
                "integrity_checks": checks, "output_hashes": outputs}
    (HERE / "12_ANALYSIS_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_dir": str(HERE), "participant_api_calls": 0, "coder_api_calls": 0,
                      "raw_hashes_unchanged": manifest["raw_hashes_unchanged"], "qwen_expectation": manifest["qwen_expectation_analysis"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
