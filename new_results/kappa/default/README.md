# Cohen's kappa between the models and the annotator groups

**This folder (section 7.2): the default prompt only** (the rating of the text as it is). The SCD prompt is in the folder `scd`.

Nothing here asks a model for anything: it reads the AU results and the labels the models already gave.

## Definitions

- **Attributed texts**: the AU rows (text x characteristic) of the final AU results, after the removal of groups in fewer than 10 AU texts (those groups stay
  removed). **Kumar** was evaluated on a sample of 65 rows, so only those rows are used for Kumar (13 groups x 10).
- **Group position** of a text: the group's label in the AU results: the majority of the binary labels of its ratings (a rating of 3 or more is toxic), an exact
  tie decided by the group's mean rating (3 or more is toxic). For **MHS** the humans' scale is 1-3, so only a rating of 3 ("Yes") is toxic, and the tie rule uses the
  mean on that scale; the models rate 1-5 and are binarized at 3 like every other dataset.
- **Model label**: the model's 1-5 rating binarized (3 or more is toxic). `baseline`: its rating of the text; `scd`: its rating of the text as a person of that group.
- **kappa**: Cohen's kappa between the model's label and the group's position over the texts in which the group is compared. A **constant model label gives kappa 0**
  and `constant_model = True` (also inside a resample). `constant_group` flags a group whose position is the same in all its texts.
- **small_n**: the characteristic has fewer than 10 attributed texts (Attitudes age and traditionalism; these characteristics stay in).
- **se**: the standard error, the standard deviation (ddof = 1) of the bootstrap values of that quantity over the valid resamples (the same resamples as the percentiles).
- **Bootstrap**: the attributed texts of a characteristic are resampled with replacement 10,000 times (seed 42); every group, model and prompt is recomputed on the same
  resamples. `ci_low` and `ci_high` are the 2.5th and 97.5th percentiles. A resample in which a group has no text has no kappa: it is counted in
  `resamples_without_value` and left out of the percentiles.

## Files

| file | content |
|---|---|
| `kappa_by_group.csv` | one row per dataset, model, characteristic, group and prompt: `n_texts`, `kappa`, `ci_low`, `ci_high`, `resamples_without_value`, `constant_model`, `constant_group`, `pct_toxic_model`, `pct_toxic_group`, `small_n`, `se`, `mirrored` (a characteristic with two groups that take opposite sides on every text) |
| `kappa_gaps.csv` | per dataset, model, characteristic, prompt and pair of groups: `gap` = kappa(A) - kappa(B), its interval from the same resamples, `share_lt0`, `share_eq0`, `share_gt0`, `share_le0` (% of resamples with a gap below 0, exactly 0, above 0, 0 or less), `interval_excludes_zero` |
| `kappa_prompt_change.csv` | per dataset, model, characteristic and group: `change` = kappa(scd) - kappa(baseline), its interval, `share_lt0`, `share_eq0`, `share_gt0`, `share_le0`, `interval_excludes_zero` |
| `constant_label_summary.csv` | how many combinations have a constant model label, per dataset, model and prompt, and in total |
