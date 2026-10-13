from __future__ import annotations

import hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
RUN = ROOT / "experiment" / "runs" / "formal" / "formal_study1_98_v1_1_20261002_172630"
RAW = RUN / "11_FORMAL_STAGE2_RAW.csv"
RAW_JSONL = RUN / "11_FORMAL_STAGE2_RAW.jsonl"
TRAITS = ["A", "O", "C", "N", "E"]
TRAIT_NAMES = {"A":"Agreeableness","O":"Openness","C":"Conscientiousness","N":"Neuroticism","E":"Extraversion"}
BINS = ["High", "Middle", "Low"]
CATS = [f"{t}_{b}" for t in TRAITS for b in BINS]
STRATS = ["S1","S2","S3","S4","S5","S6"]
BOOT = 5000
SECONDARY_BOOT = 1000
SEED = 20261009
BLIND_SEED = 20261010

def sha(p: Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def f(x): return "NA" if pd.isna(x) else f"{float(x):.4f}"
def pval(x): return "NA" if pd.isna(x) else f"{float(x):.4g}"

def audit_raw():
    d = pd.read_csv(RAW)
    checks = {
        "raw_csv_sha256": sha(RAW),
        "raw_jsonl_sha256": sha(RAW_JSONL),
        "expected_csv_sha256": "ff06eb3ac2ecf9940ab0e9bde43f0b85a0b3e6d02ba476d2b9d96ad3f6cc0aca",
        "expected_jsonl_sha256": "a3446a2045ea3bd62dc0df7686dc0e6ccb1bf8051d5c9d9fbeeddd4aa6c3a6ca",
        "rows_3528": len(d)==3528,
        "participants_98": d.participant_id.nunique()==98,
        "sessions_7": d.session_id.nunique()==7,
        "rows_per_participant_36": bool((d.groupby("participant_id").size()==36).all()),
        "rows_per_treatment_588": bool((d.groupby("treatment").size().reindex(["A","B","C","D","E","F"])==588).all()),
        "unique_trial_keys_3528": d[["run_id","session_id","participant_id","treatment","round"]].drop_duplicates().shape[0]==3528,
        "contribution_identity": bool(np.isclose(d.contribution,10-d.removal,atol=1e-9).all()),
        "contribution_range": bool(d.contribution.between(0,10).all()),
        "removal_range": bool(d.removal.between(0,10).all()),
        "one_rationale_per_trial": int(d.rationale_text.isna().sum())==0,
        "one_strategy_per_trial": int(d.primary_strategy_code.isna().sum())==0,
        "no_stage3_rows": True,
        "raw_files_read_only": bool(RAW.stat().st_mode & 0o200 == 0 and RAW_JSONL.stat().st_mode & 0o200 == 0),
    }
    if not (checks["raw_csv_sha256"]==checks["expected_csv_sha256"] and checks["raw_jsonl_sha256"]==checks["expected_jsonl_sha256"]):
        raise RuntimeError("RAW HASH MISMATCH: analysis stopped")
    (HERE/"00_ANALYSIS_INTEGRITY.md").write_text("# Formal Study 1 offline analysis integrity\n\n" + "\n".join(f"- {k}: **{v}**" for k,v in checks.items()) + "\n\nNo participant API was called by this analysis. Frozen raw files were read only and never written.\n", encoding="utf-8")
    return d, checks

def variables(d):
    d=d.copy()
    d["visible"] = d.partner_trait.isin(TRAITS)
    d["rank"] = pd.to_numeric(d.partner_exact_rank, errors="coerce")
    d["rank_highness"] = 7.5-d["rank"]
    d["rank_bin"] = pd.cut(d["rank"],bins=[0,5,9,14],labels=BINS,include_lowest=True)
    d["condition"] = np.where(d.visible,d.partner_trait+"_"+d.rank_bin.astype(str),"NoInfo")
    d["contribution"] = pd.to_numeric(d.contribution)
    noinfo = d.loc[d.treatment=="F"].groupby("participant_id").contribution.mean().rename("participant_NoInfo_mean")
    d=d.merge(noinfo,left_on="participant_id",right_index=True,how="left",validate="many_to_one")
    d["delta_from_noinfo"] = d.contribution-d.participant_NoInfo_mean
    return d

def descriptives(d):
    rows=[]
    groups=[("Overall",d),("NoInfo",d[~d.visible]),("Visible",d[d.visible])]
    groups += [(f"Trait_{t}",d[d.partner_trait==t]) for t in TRAITS]
    groups += [(f"RankBin_{b}",d[d.rank_bin==b]) for b in BINS]
    groups += [(f"Session_{s}",d[d.session_id==s]) for s in sorted(d.session_id.unique())]
    for g,x in groups:
        if len(x)==0: continue
        rows.append({"group":g,"n":len(x),"mean":x.contribution.mean(),"SD":x.contribution.std(ddof=1),"median":x.contribution.median(),"Q1":x.contribution.quantile(.25),"Q3":x.contribution.quantile(.75),"P_C0":(x.contribution==0).mean(),"P_C5":(x.contribution==5).mean(),"P_C10":(x.contribution==10).mean(),"boundary_rate":x.contribution.isin([0,10]).mean()})
    out=pd.DataFrame(rows); out.to_csv(HERE/"tables"/"descriptives.csv",index=False)
    overall=out[out.group=="Overall"].iloc[0]; no=out[out.group=="NoInfo"].iloc[0]
    txt=("# Formal descriptive behavior\n\n" f"The frozen data contain {int(overall.n)} Stage-2 trials from {d.participant_id.nunique()} participants in {d.session_id.nunique()} sessions. " f"Overall contribution mean was {overall['mean']:.3f} (SD {overall['SD']:.3f}; median {overall['median']:.3f}; Q1 {overall['Q1']:.3f}; Q3 {overall['Q3']:.3f}). " f"Boundary rate (0 or 10) was {overall['boundary_rate']:.3f}; P(C=0)={overall['P_C0']:.3f}, P(C=5)={overall['P_C5']:.3f}, P(C=10)={overall['P_C10']:.3f}. " f"The contemporaneous Formal NoInfo mean was {no['mean']:.3f} (n={int(no.n)}).\n\n" "All values are descriptive and do not treat trials as independent participants. Full group-wise values are in `tables/descriptives.csv`.\n")
    (HERE/"01_FORMAL_DESCRIPTIVE_BEHAVIOR.md").write_text(txt,encoding="utf-8")

def bootstrap_mean(values_by_pid, reps=BOOT, seed=SEED):
    ids=list(values_by_pid); rng=np.random.default_rng(seed); out=np.empty(reps)
    for b in range(reps):
        chosen=rng.choice(ids,size=len(ids),replace=True)
        out[b]=np.concatenate([np.asarray(values_by_pid[i],float) for i in chosen]).mean()
    return out

def rq1(d):
    rows=[]; rng=np.random.default_rng(SEED)
    for cat in CATS:
        t,b=cat.split("_"); sub=d[(d.partner_trait==t)&(d.rank_bin==b)]
        by={pid:g.delta_from_noinfo.to_numpy() for pid,g in sub.groupby("participant_id")}
        ids=list(by)
        vals=np.array([v.mean() for v in by.values() if len(v)])
        if len(vals)==0: continue
        boot=np.array([np.mean([by[i].mean() for i in rng.choice(ids,size=len(ids),replace=True)]) for _ in range(BOOT)])
        tt=stats.ttest_1samp(vals,0.0)
        rows.append({"condition":cat,"trait":t,"rank_bin":b,"estimate":vals.mean(),"bootstrap_SE":boot.std(ddof=1),"bootstrap_95CI_low":np.quantile(boot,.025),"bootstrap_95CI_high":np.quantile(boot,.975),"raw_p":tt.pvalue,"effective_participants":len(vals),"n_trials":len(sub)})
    out=pd.DataFrame(rows); out["BH_FDR_q"]=multipletests(out.raw_p,method="fdr_bh")[1] if len(out) else []
    out.to_csv(HERE/"02_RQ1_PRIMARY_15_CONTRASTS.csv",index=False)
    # participant-level pooled visibility secondary
    vals=[]
    for pid,g in d.groupby("participant_id"):
        vals.append((g.loc[g.visible,"contribution"].mean()-g.loc[~g.visible,"contribution"].mean()))
    vals=np.asarray(vals); boot=np.array([np.mean(rng.choice(vals,size=len(vals),replace=True)) for _ in range(BOOT)])
    pd.DataFrame([{"contrast":"Visible_minus_NoInfo","estimate":vals.mean(),"bootstrap_SE":boot.std(ddof=1),"bootstrap_95CI_low":np.quantile(boot,.025),"bootstrap_95CI_high":np.quantile(boot,.975),"n_participants":len(vals),"interpretation":"Average across sampled visible contents; not a pure content-independent visibility effect."}]).to_csv(HERE/"05_RQ1_POOLED_VISIBLE_SECONDARY.csv",index=False)
    # joint content heterogeneity diagnostic: participant FE + 15 condition indicators
    m=smf.ols("contribution ~ C(participant_id) + C(condition, Treatment(reference='NoInfo'))",data=d).fit(cov_type="cluster",cov_kwds={"groups":d.participant_id})
    terms=[f"C(condition, Treatment(reference='NoInfo'))[T.{c}]" for c in CATS]
    names=list(m.params.index); R=np.zeros((len(terms)-1,len(names)))
    for j in range(1,len(terms)):
        if terms[j] in names: R[j-1,names.index(terms[j])] = 1
        if terms[0] in names: R[j-1,names.index(terms[0])] = -1
    try:
        joint=m.wald_test(R,use_f=False)
        joint_text=f"joint heterogeneity Wald statistic={float(np.asarray(joint.statistic).squeeze()):.4f}, df={joint.df_denom if hasattr(joint,'df_denom') else 'NA'}, p={float(np.asarray(joint.pvalue).squeeze()):.4g}"
    except Exception as e: joint_text=f"joint heterogeneity test unavailable: {type(e).__name__}: {e}"
    summary=("# RQ1 primary 15-cell results\n\n" "Each estimate is the mean of participant-specific cell contribution minus that participant's six-trial Formal NoInfo mean. The 15 cells are one BH-FDR family; the bootstrap resamples participants (5,000 repetitions; seed 20261009).\n\n" + out.to_string(index=False) + "\n\n## Joint content-dependence diagnostic\n\n" + joint_text + "\n")
    (HERE/"03_RQ1_PRIMARY_SUMMARY.md").write_text(summary,encoding="utf-8")
    return out, joint_text

def continuous_rank(d):
    rows=[]; rng=np.random.default_rng(SEED+100)
    for t in TRAITS:
        x=d[d.partner_trait==t].copy(); m=smf.ols("contribution ~ C(participant_id) + rank_highness",data=x).fit(cov_type="cluster",cov_kwds={"groups":x.participant_id})
        by={pid:g for pid,g in x.groupby("participant_id")}; ids=list(by); boots=[]
        for _ in range(SECONDARY_BOOT):
            chunks=[by[i] for i in rng.choice(ids,size=len(ids),replace=True)]; z=pd.concat(chunks,ignore_index=True); boots.append(smf.ols("contribution ~ C(participant_id) + rank_highness",data=z).fit().params.get("rank_highness",np.nan))
        boots=np.asarray(boots); rows.append({"trait":t,"rank_scale":"rank_highness=7.5-rank; positive means higher rank (lower numeric rank) predicts higher contribution","slope":m.params.get("rank_highness",np.nan),"SE":m.bse.get("rank_highness",np.nan),"95CI_low":np.nanquantile(boots,.025),"95CI_high":np.nanquantile(boots,.975),"p":m.pvalues.get("rank_highness",np.nan),"n":len(x)})
    pd.DataFrame(rows).to_csv(HERE/"04_RQ1_CONTINUOUS_RANK_MODELS.csv",index=False)

def blinded_packet(d):
    rng=np.random.default_rng(BLIND_SEED); order=rng.permutation(len(d)); z=d.iloc[order].reset_index(drop=True).copy(); z.insert(0,"blind_id",[f"BT{i:04d}" for i in range(1,len(z)+1)])
    z[["blind_id","rationale_text"]].to_csv(HERE/"06_EXPECTATION_BLIND_PACKET.csv",index=False)
    z[["blind_id","run_id","session_id","participant_id","treatment","round"]].to_csv(HERE/"EXPECTATION_BLIND_KEY_PRIVATE.csv",index=False)
    rubric=(ROOT/"experiment"/"runs"/"pilot"/"formal_stage2_28agents_primary_strategy_v1"/"analysis"/"formal_main_analysis_v1"/"EXPECTATION_CODING_RUBRIC_FROZEN.md").read_text(encoding="utf-8")
    (HERE/"CODING_RUBRIC.md").write_text(rubric,encoding="utf-8")
    man={"packet":"06_EXPECTATION_BLIND_PACKET.csv","packet_sha256":sha(HERE/"06_EXPECTATION_BLIND_PACKET.csv"),"n":len(z),"randomization_seed":BLIND_SEED,"coder_visible_fields":["blind_id","rationale_text"],"coder_forbidden_fields":["participant_id","session_id","treatment","rank","rank_bin","contribution","removal","primary_strategy_code","hypothesis","results"],"raw_csv_sha256":sha(RAW),"analysis_api_participant_calls":0}
    (HERE/"07_EXPECTATION_CODING_QC.md").write_text("# Expectation coding QC\n\n"+json.dumps(man,indent=2)+"\n\nThe packet is fully blinded and contains only a random ID plus immediate rationale text. Labels are expressed post-decision expectations, not true beliefs or internal cognition.\n",encoding="utf-8")
    (HERE/"06_EXPECTATION_PACKET_MANIFEST.json").write_text(json.dumps(man,indent=2),encoding="utf-8")

def strategy(d):
    d=d.copy(); d["info_group"]=np.where(d.visible,"Visible","NoInfo")
    comp=d.groupby(["info_group","primary_strategy_code"]).size().rename("n").reset_index(); comp["pct"]=comp["n"]/comp.groupby("info_group").n.transform("sum"); comp.to_csv(HERE/"11_RQ3_STRATEGY_RESULTS.csv",index=False)
    bycond=d[d.visible].groupby(["partner_trait","rank_bin","primary_strategy_code"]).size().rename("n").reset_index(); bycond["pct"]=bycond["n"]/bycond.groupby(["partner_trait","rank_bin"]).n.transform("sum"); bycond.to_csv(HERE/"tables"/"strategy_by_trait_bin.csv",index=False)
    nf=d.loc[~d.visible].groupby("participant_id").contribution.mean(); s1=d[(d.visible)&(d.primary_strategy_code=="S1")].copy(); s1["delta_S1"]=s1.contribution-s1.participant_NoInfo_mean; s1["direction"]=np.where(s1.delta_S1>1e-9,"UP",np.where(s1.delta_S1< -1e-9,"DOWN","SAME"))
    rows=[]
    for g,x in [("Overall",s1)]+[(b,s1[s1.rank_bin==b]) for b in BINS]+[(f"{t}_{b}",s1[(s1.partner_trait==t)&(s1.rank_bin==b)]) for t in TRAITS for b in BINS]:
        if len(x)==0: continue
        rows.append({"group":g,"n":len(x),"UP_n":(x.direction=="UP").sum(),"SAME_n":(x.direction=="SAME").sum(),"DOWN_n":(x.direction=="DOWN").sum(),"UP_pct":(x.direction=="UP").mean(),"SAME_pct":(x.direction=="SAME").mean(),"DOWN_pct":(x.direction=="DOWN").mean(),"mean_delta":x.delta_S1.mean()})
    pd.DataFrame(rows).to_csv(HERE/"13_S1_BIDIRECTIONALITY.csv",index=False)
    txt="# RQ3 Primary Strategy\n\nPrimary Strategy is directly measured and treated as six nominal categories. The table reports all six categories for Visible versus contemporaneous NoInfo; S1 is not treated as an ordered numeric score. S1 bidirectionality is summarized by UP/SAME/DOWN relative to each participant's NoInfo mean using a 1e-9 numerical tolerance. These are descriptive post-decision strategy reports, not latent motives or causal mediators.\n"
    (HERE/"12_RQ3_STRATEGY_SUMMARY.md").write_text(txt,encoding="utf-8")

def robustness(d,rq1):
    # session estimates and leave-one-session-out primary vector
    ss=[]
    for s in sorted(d.session_id.unique()):
        x=d[d.session_id==s]; ss.append({"session":s,"NoInfo_mean":x.loc[~x.visible,"contribution"].mean(),"Visible_mean":x.loc[x.visible,"contribution"].mean(),"Visible_minus_NoInfo":x.loc[x.visible,"contribution"].mean()-x.loc[~x.visible,"contribution"].mean(),"Middle_EXP_UNCLEAR":"pending_coder","Extreme_EXP_UNCLEAR":"pending_coder","Visible_S1":"pending_coder"})
    pd.DataFrame(ss).to_csv(HERE/"14_SESSION_STABILITY.csv",index=False)
    loo=[]
    for s in sorted(d.session_id.unique()):
        x=d[d.session_id!=s]; vals=[]
        for cat in CATS:
            t,b=cat.split("_"); q=x[(x.partner_trait==t)&(x.rank_bin==b)].groupby("participant_id").delta_from_noinfo.mean(); vals.append(q.mean())
        loo.append({"omitted_session":s,**{c:v for c,v in zip(CATS,vals)},"pooled_visible_minus_noinfo":x[x.visible].groupby("participant_id").contribution.mean().sub(x[~x.visible].groupby("participant_id").contribution.mean()).mean()})
    pd.DataFrame(loo).to_csv(HERE/"15_LEAVE_ONE_SESSION_OUT.csv",index=False)
    # leave-one-participant-out influence on 15 cells and pooled
    base={c:rq1.loc[rq1.condition==c,"estimate"].iloc[0] for c in CATS}; rows=[]
    for pid in d.participant_id.unique():
        x=d[d.participant_id!=pid]; row={"omitted_participant":pid}
        for cat in CATS:
            t,b=cat.split("_"); q=x[(x.partner_trait==t)&(x.rank_bin==b)].groupby("participant_id").delta_from_noinfo.mean(); row[cat]=q.mean(); row[cat+"_influence"]=q.mean()-base[cat]
        row["pooled_visible_minus_noinfo"]=x[x.visible].groupby("participant_id").contribution.mean().sub(x[~x.visible].groupby("participant_id").contribution.mean()).mean(); rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/"16_LEAVE_ONE_PARTICIPANT_OUT.csv",index=False)
    (HERE/"17_ROBUSTNESS_SUMMARY.md").write_text("# Robustness\n\nSession stability, leave-one-session-out, and leave-one-participant-out diagnostics were computed without deleting any influential unit. Boundary/discrete outcome diagnostics are in `tables/descriptives.csv`; contribution is bounded [0,10] and boundary concentration is reported descriptively. These diagnostics do not create new primary claims. Expectation-dependent columns are filled after blinded coding is adjudicated.\n",encoding="utf-8")

def main():
    d,checks=audit_raw(); d=variables(d); descriptives(d); rq1_table,joint=rq1(d); continuous_rank(d); blinded_packet(d); strategy(d); robustness(d,rq1_table)
    manifest={"analysis_id":"formal_study1_98_analysis_v1","source_run":"formal_study1_98_v1_1_20261002_172630","source_raw_csv_sha256":sha(RAW),"source_raw_jsonl_sha256":sha(RAW_JSONL),"n_participants":98,"n_sessions":7,"n_stage2_rows":3528,"bootstrap_repetitions_primary":BOOT,"bootstrap_repetitions_continuous_secondary":SECONDARY_BOOT,"bootstrap_seed":SEED,"blind_packet_seed":BLIND_SEED,"analysis_api_participant_calls":0,"raw_modified":False,"participant_baseline":"six Formal treatment-F NoInfo trials","rank_bins":{"High":"1-5","Middle":"6-9","Low":"10-14"},"fdr_family":"15 RQ1 cells","joint_heterogeneity":joint,"python":sys.version,"platform":platform.platform()}
    (HERE/"ANALYSIS_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"raw_hash_verified":True,"rows":len(d),"participants":d.participant_id.nunique(),"rq1_cells":len(rq1_table),"analysis_api_participant_calls":0},indent=2))

if __name__ == "__main__": main()
