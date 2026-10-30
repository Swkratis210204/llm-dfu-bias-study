"""Pooled kappa vs. group kappa (new analysis; changes nothing that exists).

For every model and characteristic: Cohen's kappa of the model's DEFAULT-prompt label against the POOLED label (all ratings of a text after the
preprocessing) next to the kappa against each group's position, on the attributed (AU) texts of the paper's alignment tables (Table 4, 16-18).

Everything is reused from ``perspective_audit.kappa_groups`` (attributed rows, model labels, per-characteristic arrays, the vectorised kappa, the
resampling with seed 42 and 10,000 resamples) and ``Unimodality.label_info`` (the label rule: majority, an exact tie decided by the mean).
The group kappas are checked against ``new_results/kappa/default/kappa_by_group.csv``; any difference stops the script before anything is written.

Run from the repository root:  python scripts/pooled_vs_group.py
Output: results/pooled_vs_group/pooled_vs_group.csv, summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perspective_audit" / "src"))     # works without installing the package

from perspective_audit import find_root, load_results
from perspective_audit.eda_total import _ordered
from perspective_audit.kappa_groups import (MODELS, _attributed_rows, _block, _interval, _kappa, _model_labels, _r, _se)
from perspective_audit.persona_annotation import collect_persona_runs

REPS, SEED = 10_000, 42
MODEL_NAME = {"chatgpt": "ChatGPT", "deepseek": "DeepSeek", "gemini": "Gemini", "mistral": "Mistral"}
DAGGER = "†"


def pooled_labels(result) -> pd.Series:
    """item id (str) -> (pooled label as 0/1, decided by the mean) from ALL the ratings of the text in the cleaned data, same rule as a group position."""
    out = {}
    for item, ratings in result.clean.groupby("item_id")["rating"]:
        label, by_mean = result.tool.label_info(ratings.to_numpy())
        out[str(item)] = (1.0 if label == "toxic" else 0.0, by_mean)
    return pd.Series(out)


def _row(dataset, dimension, model, kind, group, n, point, boot, constant, share=np.nan):
    low, high, _ = _interval(boot)
    return {"dataset": dataset, "characteristic": dimension, "model": model, "row_type": kind, "group": group, "n_texts": int(n), "kappa": _r(point),
            "se": _r(_se(boot)), "ci_low": _r(low), "ci_high": _r(high), "constant_flag": bool(constant), "share_pooled_equals_group": share}


def compute(results, results_dir: Path, given: pd.DataFrame):
    scd = collect_persona_runs(results_dir, "scd")               # only needed because _block checks the persona answers too
    rows, tie_rows = [], []
    for dataset in _ordered(results):
        result = results[dataset]
        attributed = _attributed_rows(result, given, dataset)
        labels = _model_labels(results_dir, dataset, MODELS, scd)
        pooled = pooled_labels(result)
        for dimension in sorted(attributed["dimension"].unique(), key=list(result.spec.dimensions).index):
            block = _block(attributed, dimension, labels, MODELS, dataset)
            part = attributed[attributed["dimension"] == dimension].reset_index(drop=True)
            info = pooled.loc[part["item_id"]]
            position_pooled = np.array([v[0] for v in info])
            tie_rows.append({"dataset": dataset, "characteristic": dimension, "attributed texts": len(part), "pooled ties resolved by the mean": int(sum(v[1] for v in info)),
                             "pooled toxic": int(position_pooled.sum()), "pooled civil": int(len(part) - position_pooled.sum())})
            n = block["n"]
            counts = np.random.default_rng(SEED).multinomial(n, np.full(n, 1.0 / n), size=REPS).astype(float)       # the resamples of the existing tables
            everything = np.ones((1, n))
            for model in MODELS:
                label = block["baseline"][model]
                rows.append(_row(dataset, dimension, model, "pooled", "pooled", n, _kappa(everything, position_pooled, label)[0],
                                 _kappa(counts, position_pooled, label), len(set(label)) == 1))
                for j, group in enumerate(block["groups"]):
                    present, position = block["present"][:, j], block["position"][:, j]
                    shown = present.astype(bool)
                    share = round(100 * float((position_pooled[shown] == position[shown]).mean()), 2)
                    constant = len(set(label[shown])) == 1
                    rows.append(_row(dataset, dimension, model, "group", group, present.sum(), _kappa(present[None, :], position, label)[0],
                                     _kappa(counts * present[None, :], position, label), constant, share))
                    rows.append(_row(dataset, dimension, model, "pooled_on_group_subset", group, present.sum(), _kappa(present[None, :], position_pooled, label)[0],
                                     _kappa(counts * present[None, :], position_pooled, label), constant, share))
    return pd.DataFrame(rows), pd.DataFrame(tie_rows)


def check_against_paper(table: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """The group rows against kappa_by_group.csv (default prompt): kappa, n, se and interval must be identical. Returns the rows that differ."""
    old = pd.read_csv(existing_path)
    old = old[old["prompt"] == "baseline"].rename(columns={"model": "model"})
    mine = table[table["row_type"] == "group"]
    merged = mine.merge(old, on=["dataset", "characteristic", "model", "group"], how="outer", suffixes=("", "_paper"), indicator=True)
    bad = merged[(merged["_merge"] != "both") | ~np.isclose(merged["kappa"], merged["kappa_paper"], atol=1e-9) | (merged["n_texts"] != merged["n_texts_paper"])
                 | ~np.isclose(merged["se"], merged["se_paper"], atol=1e-9) | ~np.isclose(merged["ci_low"], merged["ci_low_paper"], atol=1e-9)
                 | ~np.isclose(merged["ci_high"], merged["ci_high_paper"], atol=1e-9) | (merged["constant_flag"] != merged["constant_model"])]
    return bad[["dataset", "characteristic", "model", "group", "n_texts", "n_texts_paper", "kappa", "kappa_paper", "se", "se_paper", "ci_low", "ci_low_paper",
                "ci_high", "ci_high_paper", "_merge"]]


def _fmt(row) -> str:
    return f"{row['kappa']:.2f}{DAGGER if row['constant_flag'] else ''}"


def dataset_tables(table: pd.DataFrame, dataset: str) -> str:
    out = []
    part = table[table["dataset"] == dataset]
    for dimension in dict.fromkeys(part["characteristic"]):
        sub = part[part["characteristic"] == dimension]
        groups = [g for g in dict.fromkeys(sub[sub["row_type"] == "group"]["group"])]
        n_groups = {g: int(sub[(sub["row_type"] == "group") & (sub["group"] == g)]["n_texts"].iloc[0]) for g in groups}
        n_pooled = int(sub[sub["row_type"] == "pooled"]["n_texts"].iloc[0])
        header = [f"pooled (n={n_pooled})"] + [f"{g} (n={n_groups[g]})" for g in groups]
        out += [f"**{dimension}**", "", "| model | " + " | ".join(header) + " |", "|---|" + "---|" * len(header)]
        for model in MODELS:
            cells = [_fmt(sub[(sub["model"] == model) & (sub["row_type"] == "pooled")].iloc[0])]
            cells += [_fmt(sub[(sub["model"] == model) & (sub["row_type"] == "group") & (sub["group"] == g)].iloc[0]) for g in groups]
            out.append(f"| {MODEL_NAME[model]} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def counts(table: pd.DataFrame):
    """Per (dataset, characteristic, model): pooled kappa strictly between the lowest and highest group kappa; |pooled| < 0.2 with group kappas >= 0.5 apart."""
    rows = []
    for (dataset, dimension, model), part in table.groupby(["dataset", "characteristic", "model"], sort=False):
        pooled = part[part["row_type"] == "pooled"]["kappa"].iloc[0]
        group = part[part["row_type"] == "group"]["kappa"]
        low, high = group.min(), group.max()
        rows.append({"dataset": dataset, "characteristic": dimension, "model": model, "pooled": pooled, "min_group": low, "max_group": high, "spread": round(high - low, 4),
                     "between": bool(low < pooled < high), "hidden": bool(abs(pooled) < 0.2 and high - low >= 0.5 - 1e-9)})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame, ties: pd.DataFrame, cases: pd.DataFrame, rows_used: pd.DataFrame) -> str:
    L = ["# Pooled kappa vs. group kappa (default prompt)", "",
         f"Cohen's kappa of the model's default-prompt label against the pooled label of the attributed texts and against each group's position. Bootstrap: {REPS:,} resamples of the "
         f"texts, seed {SEED} (those of the existing tables). {DAGGER} = the model gave one label to every text (kappa set to 0). Group kappas are identical to Table 4 and Tables 16-18 "
         "(checked by the script). Kumar: the 65-attribution sample.", ""]
    L += ["## 1. Kappa by dataset and characteristic", ""]
    for dataset in _ordered(table["dataset"].unique()):
        L += [f"### {dataset}", "", dataset_tables(table, dataset)]
    L += ["## 2. Kumar gender", "", "| model | pooled | female | male |", "|---|---|---|---|"]
    k = table[(table["dataset"] == "kumar") & (table["characteristic"] == "gender")]
    for model in MODELS:
        m = k[k["model"] == model]
        cells = [_fmt(m[m["row_type"] == "pooled"].iloc[0])] + [_fmt(m[(m["row_type"] == "group") & (m["group"].str.lower() == g)].iloc[0]) for g in ("female", "male")]
        L.append(f"| {MODEL_NAME[model]} | " + " | ".join(cells) + " |")
    pooled_k = k[k["row_type"] == "pooled"]
    L += ["", "Pooled kappa with its bootstrap SD and 2.5-97.5 percentile range:", "", "| model | pooled kappa | SD | 2.5% | 97.5% |", "|---|---|---|---|---|"]
    for model in MODELS:
        r = pooled_k[pooled_k["model"] == model].iloc[0]
        L.append(f"| {MODEL_NAME[model]} | {r['kappa']:.2f}{DAGGER if r['constant_flag'] else ''} | {r['se']:.2f} | {r['ci_low']:.2f} | {r['ci_high']:.2f} |")
    L += ["", "## 3. Counts over all datasets (model x characteristic cases; every characteristic has at least two groups)", "",
          f"- cases: {len(cases)}", f"- pooled kappa strictly between the lowest and highest group kappa: {int(cases['between'].sum())}",
          f"- |pooled| < 0.2 while the group kappas differ by >= 0.5: {int(cases['hidden'].sum())}", "", "| dataset | cases | pooled strictly between | |pooled| < 0.2 and spread >= 0.5 |", "|---|---|---|---|"]
    for dataset, c in cases.groupby("dataset", sort=False):
        L.append(f"| {dataset} | {len(c)} | {int(c['between'].sum())} | {int(c['hidden'].sum())} |")
    L += ["", "## 4. Caveats and checks", "",
          "- Pooled label: all ratings of the text in the cleaned data (all annotators, also those of groups removed by the rule of ten AU texts; ratings with a removed value in any "
          "characteristic were dropped by the preprocessing, so the same pooled label is used for every characteristic of a text). Toxic if most ratings are >= 3 (MHS: only 3), "
          "an exact tie decided by the mean (>= 3 toxic).", "",
          "Pooled ties resolved by the mean, per dataset (distinct attributed texts / attributed rows = text x characteristic):", "",
          "| dataset | distinct texts | ties (texts) | rows | ties (rows) |", "|---|---|---|---|---|"]
    for _, r in rows_used.iterrows():
        L.append(f"| {r['dataset']} | {r['texts']} | {r['tie_texts']} | {r['rows']} | {r['tie_rows']} |")
    L += ["", "Per characteristic:", "", "| dataset | characteristic | attributed texts | ties by the mean | pooled toxic | pooled civil |", "|---|---|---|---|---|---|"]
    for _, r in ties.iterrows():
        L.append(f"| {r['dataset']} | {r['characteristic']} | {r['attributed texts']} | {r['pooled ties resolved by the mean']} | {r['pooled toxic']} | {r['pooled civil']} |")
    const = table[table["constant_flag"] & (table["row_type"] == "pooled")]
    L += ["", f"- Pooled rows with a constant model label (kappa 0, {DAGGER}): {len(const)} of {int((table['row_type'] == 'pooled').sum())}.",
          f"- Group rows with a constant model label: {int(table[table['row_type'] == 'group']['constant_flag'].sum())} of {int((table['row_type'] == 'group').sum())}.",
          "- Cases where the pooled label is the same for every attributed text of the characteristic (pooled kappa is then 0 by construction): "
          + (", ".join(f"{r['dataset']}/{r['characteristic']}" for _, r in ties.iterrows() if r["pooled toxic"] in (0, r["attributed texts"])) or "none") + ".",
          "- Datasets where most pooled labels are decided by the mean (exact tie between toxic and civil ratings): "
          + (", ".join(f"{r['dataset']} ({r['tie_texts']} of {r['texts']} texts)" for _, r in rows_used.iterrows() if r["tie_texts"] > r["texts"] / 2) or "none")
          + ". There the pooled label is mostly a mean-threshold decision.",
          "- Share of texts where the pooled label equals the group position: column `share_pooled_equals_group` of the csv.", "- Match with the paper: see the line below.", ""]
    return "\n".join(L)


def main():
    root = find_root()
    results = load_results()
    given = pd.read_csv(root / "new_results" / "aposteriori_for_models.csv", dtype={"item_id": str})
    table, ties = compute(results, root / "new_results", given)
    bad = check_against_paper(table, root / "new_results" / "kappa" / "default" / "kappa_by_group.csv")
    if len(bad):
        print("MISMATCH with the paper's tables; nothing written:")
        print(bad.to_string())
        sys.exit(1)
    print(f"group rows identical to new_results/kappa/default/kappa_by_group.csv ({int((table['row_type'] == 'group').sum())} rows)")
    cases = counts(table)
    rows_used = []
    for dataset in _ordered(results):
        attributed = _attributed_rows(results[dataset], given, dataset)
        pooled = pooled_labels(results[dataset])
        flags = pooled.loc[attributed["item_id"]].map(lambda v: v[1])
        distinct = attributed["item_id"].drop_duplicates()
        rows_used.append({"dataset": dataset, "texts": len(distinct), "tie_texts": int(pooled.loc[distinct].map(lambda v: v[1]).sum()), "rows": len(attributed), "tie_rows": int(flags.sum())})
    out = root / "results" / "pooled_vs_group"
    out.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "characteristic", "model", "row_type", "group", "n_texts", "kappa", "se", "ci_low", "ci_high", "constant_flag", "share_pooled_equals_group"]
    table[columns].to_csv(out / "pooled_vs_group.csv", index=False, encoding="utf-8")
    text = summary(table, ties, cases, pd.DataFrame(rows_used))
    text = text.replace("- Match with the paper: see the line below.", "- Match with the paper: all group rows (kappa, n, SE, interval, constant flag) are identical to `new_results/kappa/default/kappa_by_group.csv` (Tables 4, 16-18).")
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
