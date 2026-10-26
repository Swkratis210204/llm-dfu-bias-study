#!/usr/bin/env bash
# Re-run the whole thesis notebook from start to end.
#
#   ./run_all.sh          full rerun: rebuild the AU tables from the data, send the texts to
#                         the six LLMs (costs API credits, needs keys in .env), recompute all
#                         Cohen's kappa / Pearson results
#   ./run_all.sh cached   no API calls: reuse the cached scores and frozen table in results/<dataset>/
#
# Outputs (tables, figures, regenerated scores, executed notebook) go to results/<dataset>/ (default: attitudes), overwriting the files there.
set -euo pipefail

MODE="${1:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB="$ROOT/Thesis/Code/ThesisCodef.ipynb"
export THESIS_DATASET="${THESIS_DATASET:-attitudes}"   # everything lives in results/<dataset>
OUT="$ROOT/results/$THESIS_DATASET"
PYTHON="${PYTHON:-/c/Python314/python}"

export PATH="$(dirname "$PYTHON"):$PATH"   # pip and the kernel below use this interpreter
export THESIS_ROOT="$ROOT"
export PYTHONIOENCODING=utf-8

"$PYTHON" -m pip install --quiet seaborn openpyxl

"$PYTHON" -m ipykernel install --user --name thesis-run --display-name "thesis-run" >/dev/null

case "$MODE" in
  full)
    export THESIS_RUN_LLMS=1        # RUN_LLM_QUERIES: call the six LLM APIs
    export THESIS_REGEN_TABLE=1     # USE_REGENERATED_PCP_TABLE: use the table rebuilt by section 6
    [ -f "$ROOT/.env" ] || { echo "Missing $ROOT/.env (copy .env.example and fill in the keys)"; exit 1; }
    "$PYTHON" -m pip install --quiet groq openai google-genai oci
    ;;
  cached)
    export THESIS_RUN_LLMS=0
    export THESIS_REGEN_TABLE=0
    ;;
  *) echo "usage: $0 [full|cached]"; exit 1 ;;
esac

for f in "$ROOT/datasets/popquorn/popquorn.csv" "$ROOT/Thesis/Code/attitudes_embedded.csv"; do
  [ -f "$f" ] || { echo "Missing input: $f"; exit 1; }
done

echo "Mode: $MODE  (LLMS=$THESIS_RUN_LLMS, REGEN_TABLE=$THESIS_REGEN_TABLE)"
mkdir -p "$OUT"
"$PYTHON" -m jupyter nbconvert --to notebook --execute "$NB" \
  --output ThesisCodef.executed.ipynb --output-dir "$OUT" \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=thesis-run

echo "Done. Executed notebook and results: $OUT"
