from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUN = PROJECT / "experiment/runs/formal/gpt56terra98_cross_model_formal_v1_20261010_220000"
RAW = RUN / "GPT56TERRA_STAGE2_RAW.csv"
FINAL = HERE / "GPT56TERRA_EXPECTATION_FINAL.csv"
OUT = HERE / "04_GPT56TERRA_EXPECTATION_DISTRIBUTION.csv"
A_OUT = HERE / "06_GPT56TERRA_AGREEABLENESS_EXPECTATION.csv"

CATS = ["EXP_COOP", "EXP_UNCOOP", "EXP_UNCLEAR", "EXP_NONE"]
TRAITS = ["A", "O", "C", "N", "E"]
BINS = ["High", "Middle", "Low"]
TREATMENT_TO_TRAIT = {"A": "A", "B": "O", "C": "C", "D": "N", "E": "E", "F": "NoInfo"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    raw = pd.read_csv(RAW, dtype=str).fillna("")
    labels = pd.read_csv(FINAL, dtype=str).fillna("")
    keys = ["participant_id", "treatment", "round"]
    if len(raw) != 3528 or raw[keys].duplicated().any():
        raise RuntimeError("frozen Terra Stage2 key structure failed")
    if len(labels) != 3528 or labels.blind_id.duplicated().any():
        raise RuntimeError("frozen Terra expectation table failed")
    if not labels.expectation_code.isin(CATS).all():
        raise RuntimeError("unresolved expectation labels remain")
    if labels[keys].duplicated().any():
        raise RuntimeError("expectation key duplicated")

    d = raw.merge(labels[keys + ["expectation_code", "final_source", "final_confidence"]], on=keys, validate="one_to_one")
    d["visible"] = (d["treatment"] != "F").astype(int)
    d["trait"] = d["treatment"].map(TREATMENT_TO_TRAIT)
    d["rank_bin"] = d["rank_bin"].where(d["visible"].eq(1), "NoInfo")

    rows = []
    scopes = [("Overall", d, "ALL", "ALL"), ("NoInfo", d[d.visible.eq(0)], "NoInfo", "NoInfo")]
    for trait in TRAITS:
        for b in BINS:
            scopes.append(("Visible", d[(d.visible.eq(1)) & (d.trait.eq(trait)) & (d.rank_bin.eq(b))], trait, b))
    for scope, z, trait, b in scopes:
        n = len(z)
        for cat in CATS:
            count = int(z.expectation_code.eq(cat).sum())
            rows.append({"model": "GPT-5.6 Terra", "scope": scope, "cell": "Overall" if scope == "Overall" else ("NoInfo" if scope == "NoInfo" else f"{trait}_{b}"), "trait": trait, "rank_bin": b, "expectation_code": cat, "count": count, "n": n, "proportion": count / n if n else None})
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT, index=False, encoding="utf-8-sig")

    a = d[(d.visible.eq(1)) & (d.trait.eq("A"))]
    arows = []
    for b in BINS:
        z = a[a.rank_bin.eq(b)]
        n = len(z)
        for cat in CATS:
            c = int(z.expectation_code.eq(cat).sum())
            arows.append({"model": "GPT-5.6 Terra", "trait": "A", "rank_bin": b, "expectation_code": cat, "count": c, "n": n, "proportion": c / n if n else None})
    aw = pd.DataFrame(arows)
    piv = aw.pivot(index="rank_bin", columns="expectation_code", values="proportion").reindex(BINS).reset_index()
    for cat in CATS:
        if cat not in piv:
            piv[cat] = 0.0
    piv["directional_coop_minus_uncoop"] = piv["EXP_COOP"] - piv["EXP_UNCOOP"]
    piv["coop_minus_uncoop_low_minus_higher"] = None
    higher = piv.loc[piv.rank_bin.isin(["High", "Middle"]), "directional_coop_minus_uncoop"].mean()
    low_i = piv.index[piv.rank_bin.eq("Low")]
    if len(low_i):
        piv.loc[low_i, "coop_minus_uncoop_low_minus_higher"] = piv.loc[low_i, "directional_coop_minus_uncoop"].iloc[0] - higher
    piv.to_csv(A_OUT, index=False, encoding="utf-8-sig")

    summary = {
        "status": "COMPLETE_GPT56TERRA_EXPECTATION_SUMMARY",
        "raw_stage2_sha256": sha256(RAW),
        "expectation_final_sha256": sha256(FINAL),
        "rows": len(d),
        "category_counts": d.expectation_code.value_counts().reindex(CATS, fill_value=0).to_dict(),
        "coder_sources": d.final_source.value_counts().to_dict(),
        "overall_distribution_path": OUT.name,
        "agreeableness_distribution_path": A_OUT.name,
        "manuscript_modified": False,
        "figures_modified": False,
    }
    (HERE / "GPT56TERRA_EXPECTATION_SUMMARY_MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
