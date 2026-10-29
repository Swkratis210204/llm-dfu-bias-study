# Prompt-wording ablation (6.2)

**Question.** Does rewording the question change the ratings of the models? The baseline prompt and two rewordings of it (R1 and R2) were run on the same random
sample of texts of every dataset; this folder compares the answers. The prompts are in `new_results/prompts.md`.

## Design

| | |
|---|---|
| What changed | the wording of the question only |
| What stayed fixed | the task, the 1-5 scale, the rule (1-2 civil, 3-5 toxic), the reply format, the texts, the four models, temperature 0; no definition, example or role was added |
| Prompts | `baseline` (run `baseline`), R1 (run `prompt_sensitivity`), R2 (run `prompt_sensitivity_2`); ten numbered texts per call |
| Sample | a random sample per dataset with seed 42, the same texts for every model and prompt: Attitudes 41, POPQUORN 50, MHS 50, Kumar 50 texts (Attitudes has only 41) |
| Models | ChatGPT `gpt-4o-mini`, DeepSeek `deepseek-chat`, Mistral `open-mistral-7b`, Gemini `gemini-2.5-flash`; all runs on 2026-10-27, the exact snapshot was not recorded |
| Runs | `new_results/run_log.csv`; the replies are in `new_results/<dataset>/<run>/raw/` |
| Each run once | no repetition of the same prompt, so the differences between R1 and R2 are the closest measure of the normal variation between wordings |

## Measures

- **same label %**: the share of (text, model) pairs with the same label (toxic, civil) under the two prompts.
- **Cohen's kappa (labels)**: the agreement of the labels corrected for chance. It is **unreliable when almost every label is the same** (MHS): it can be near 0
  with 95% of labels equal, and it is undefined when both runs give one and the same label to every text.
- **weighted kappa (scores)**: the agreement of the 1-5 scores with quadratic weights (a big disagreement counts more than a small one). 1 is perfect, 0 is chance.
- **Spearman**, **same score %** and **mean diff** (second prompt minus first; above 0 means the second prompt scores higher).
- **Intervals**: 95% percentile bootstrap, 2000 resamples of the texts (the models of a text move together in the pooled rows).

## Files

| file | content |
|---|---|
| `agreement_pooled.csv` | per dataset and pair of prompts, the four models together (12 rows) |
| `agreement_by_model.csv` | per dataset, model and pair of prompts (48 rows) |
| `wording_vs_model.csv` | a model against itself under another wording, against different models under the same prompt |
| `ablation_sample.csv` | the item ids of the sample |
| `ablation_weighted_kappa.pdf/.png`, `ablation_label_agreement.pdf/.png` | the agreement per dataset and model, with intervals |
| `ablation_wording_vs_model.pdf/.png` | a change of wording against a change of model |

## Summary of the numbers

- Pooled over the four models, the share of identical labels is 85.4% to 98.5% over the 12 rows of `agreement_pooled.csv`,
  and the weighted kappa of the scores 0.68 to 0.92. Against the baseline (R1 and R2 together) the labels agree
  91.4% on average and the weighted kappa is 0.81 on average.
- Mean weighted kappa of a model with itself under another wording (baseline vs R1 and vs R2): 0.63
  to 0.88; of different models under the baseline prompt: 0.49
  to 0.70.

## Be careful

- The samples are small (41-50 texts per dataset), so the intervals are wide, especially per model.
- Only rewordings that add no information were tested; a definition of toxicity, examples or a role were not.
- Each wording was run once. Hosted models can differ between days even at temperature 0, and the exact model snapshot was not recorded.
- **MHS:** almost every label is the same class, so the share of equal labels is high (94-100%) while Cohen's kappa is near 0 or undefined for some models; read the
  weighted kappa and the share of equal labels there.
- R2 gives slightly higher scores than the baseline in some datasets (see `mean diff` in the tables): a systematic shift, not only noise.
