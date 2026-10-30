"""Permutation null for the attribution rates and the demographic / non-demographic gap (section 7.7). The logic is in ``perspective_audit/shuffling_robustness.py``.

Run from the repository root:  python new_results/shuffling_robustness/shuffling_robustness.py
Output (next to this script): attribution_null_per_characteristic.csv, attribution_null_gap.csv, per_pair_chance_probability.csv, summary.md
1,000 permutations per characteristic (200 for Kumar only if 1,000 would take more than 30 minutes; the summary says which).
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "perspective_audit" / "src"))     # works without installing the package

from perspective_audit import compute_null, load_results, save_null

if __name__ == "__main__":
    tables = compute_null(load_results(), permutations=1_000, reps=10_000, seed=42)
    print(tables["checks"].to_string(index=False))
    print(save_null(tables, Path(__file__).resolve().parent, reps=10_000, seed=42))
