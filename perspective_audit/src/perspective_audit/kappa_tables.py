"""LaTeX tables of the kappa between the models and the groups (section 7.2): one table per dataset, from ``kappa_by_group``.

The tables need the packages ``booktabs``, ``makecell`` and ``acro`` (the macros ``\\ac{AU}``, ``\\ac{LLM}``, ``\\acp{LLM}`` and ``\\ac{SCD}`` must be defined).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from .eda_total import DATASET_LABEL

MODEL_NAME = {"chatgpt": "ChatGPT", "deepseek": "DeepSeek", "gemini": "Gemini", "mistral": "Mistral"}
PROMPT_NAME = {"baseline": "Baseline", "scd": "SCD"}
TOXIC_SHARE_DATASETS = ("mhs", "attitudes")                  # the datasets where the models give almost one label to every text
GROUP_NAME = {"millenial": "millennial"}                    # the spelling of the data is corrected in the tables

CAPTION = ("Cohen's kappa between each \\ac{LLM} and the groups of each characteristic on polarized texts attributed to it, with the baseline and the \\ac{SCD} "
           "prompt. Values in parentheses are bootstrap standard errors and values in brackets 95\\% intervals from 10,000 resamples of the attributed texts.")


def _number(value: float) -> str:
    """Two decimals, a minus sign in math mode, no ``-0.00``."""
    shown = round(float(value), 2)
    return "0.00" if shown == 0 else (f"$-${abs(shown):.2f}" if shown < 0 else f"{shown:.2f}")


def _cell(row: pd.Series) -> str:
    """``0.13 (0.11)`` over ``[-0.09, 0.37]``; a dagger after the value when the model gave one label to every text."""
    if row is None:
        return "--"
    dagger = "$^\\dagger$" if row["constant_model"] else ""
    # the interval is wrapped in braces: a bracket at the start of a line after \\ would be read as an optional length
    return f"\\makecell{{{_number(row['kappa'])}{dagger} ({_number(row['se'])})\\\\ {{[}}{_number(row['ci_low'])}, {_number(row['ci_high'])}{{]}}}}"


def _name(text: str) -> str:
    text = GROUP_NAME.get(text, text)
    return text.replace("_", "\\_")


def _characteristic(name: str, kind: str) -> str:
    return f"{name.replace('_', ' ').capitalize()} ({'D' if kind == 'demographic' else 'N'})"


def dataset_table(part: pd.DataFrame, dataset: str, models: Sequence[str]) -> str:
    """The LaTeX table (as text) of one dataset. ``part`` is the ``kappa_by_group`` table of that dataset."""
    prompts = list(PROMPT_NAME)
    models = [m for m in models if m in set(part["model"])]          # only the models that rated this dataset
    columns = 2 + len(prompts) * len(models)
    lookup = {(r.characteristic, r.group, r.model, r.prompt): r for _, r in part.iterrows()}
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             f"\\caption{{{CAPTION} Dataset: {DATASET_LABEL.get(dataset, dataset)}.}}", f"\\label{{tab:kappa_{dataset}}}",
             f"\\begin{{tabular}}{{@{{}}lr*{{{len(prompts) * len(models)}}}{{c}}@{{}}}}", "\\toprule"]
    lines.append("& & " + " & ".join(f"\\multicolumn{{{len(models)}}}{{c}}{{{PROMPT_NAME[p]}}}" for p in prompts) + " \\\\")
    lines.append("".join(f"\\cmidrule(lr){{{3 + i * len(models)}-{2 + (i + 1) * len(models)}}}" for i in range(len(prompts))))
    lines.append("Group & $n$ & " + " & ".join(MODEL_NAME.get(m, m) for _ in prompts for m in models) + " \\\\")
    lines.append("\\midrule")
    first = True
    for characteristic in part["characteristic"].drop_duplicates():
        block = part[part["characteristic"] == characteristic]
        if not first:
            lines.append("\\addlinespace")
        first = False
        lines.append(f"\\multicolumn{{{columns}}}{{@{{}}l}}{{\\textit{{{_characteristic(characteristic, block['type'].iloc[0])}}}}} \\\\")
        for group in block["group"].drop_duplicates():
            rows = block[block["group"] == group]
            small = "$^{\\ddagger}$" if bool(rows["small_n"].iloc[0]) else ""
            cells = [_cell(lookup.get((characteristic, group, m, p))) for p in prompts for m in models]
            lines.append(f"\\quad {_name(group)}{small} & {int(rows['n_texts'].iloc[0])} & " + " & ".join(cells) + " \\\\")
    if dataset in TOXIC_SHARE_DATASETS:
        lines.append("\\midrule")
        share = part.groupby(["model", "prompt"])["pct_toxic_model"].mean()
        cells = [f"{share[(m, p)]:.0f}\\%" for p in prompts for m in models]
        lines.append("\\multicolumn{2}{@{}l}{Toxic labels (\\%)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip", "\\begin{minipage}{\\linewidth}", "\\footnotesize"]
    notes = ["D = demographic, N = non-demographic characteristic; $n$ = number of attributed (\\ac{AU}) texts in which the group is compared.",
             "$^\\dagger$ the model gave the same label to every text, so kappa is 0 by definition.",
             ]
    if part["small_n"].any():
        notes.append("$^{\\ddagger}$ the characteristic has fewer than 10 attributed texts.")
    notes.append("Bounds at exactly 0.00 can come from resamples in which the model gave one label to every text, for which kappa is 0 by definition.")
    if "mirrored" in part and part["mirrored"].any():
        notes.append("For two-group characteristics the two kappas are opposite because the groups take opposite sides on every text.")
    if dataset in TOXIC_SHARE_DATASETS:
        notes.append("The last row is the share of texts that the \\acp{LLM} labeled toxic (rating of 3 or more), averaged over the groups.")
    lines += [" \\\\\n".join(notes), "\\end{minipage}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def save_kappa_tables(by_group: pd.DataFrame, folder: Path, models: Sequence[str] = tuple(MODEL_NAME)) -> Dict[str, Path]:
    """Write ``table_<dataset>.tex`` for every dataset of ``by_group`` into ``folder`` (created if needed). Returns the files, by dataset."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for dataset in by_group["dataset"].drop_duplicates():
        path = folder / f"table_{dataset}.tex"
        path.write_text(dataset_table(by_group[by_group["dataset"] == dataset], dataset, models), encoding="utf-8")
        written[dataset] = path
    return written
