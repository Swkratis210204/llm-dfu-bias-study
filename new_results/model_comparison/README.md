# Agreement between the models (section 7.4)

A diagnostic: do the models rate the same texts alike? Nothing here asks a model for anything: it reads the ratings they gave with the default prompt
(`<dataset>/baseline/<model>.csv`, the texts given to the models: for Kumar the sample).

| file | content |
|---|---|
| `pairwise.csv` | one row per dataset and pair of models: `n_texts` (rated by both), `spearman`, `kappa`, `agreement_pct`, `pct_toxic_a`, `pct_toxic_b`, `constant_a`, `constant_b` |
| `matrices.csv` | the same numbers as one matrix per dataset and measure (the heatmaps): rows and columns are the models, the diagonal is 1 |
| `model_agreement.pdf`, `.png` | the heatmaps: one column per dataset, Spearman on the top row and Cohen's kappa on the bottom row |

- **Spearman**: the rank correlation of the 1-5 scores of the two models (do they put the texts in the same order?). It has no value when a model gives the same score to every text.
- **Cohen's kappa**: the agreement of the two binary labels (a score of 3 or more is toxic) beyond chance. It is 0 when only one of the models gives one label to every text,
  and has no value when both give the same single label to every text.
- **agreement_pct**: the share of texts with the same label.
- The diagonal of the heatmaps is 1 by definition. A cell without a value is shown as `n/a`.
- No confidence intervals: this is a diagnostic of the models, not a test.
