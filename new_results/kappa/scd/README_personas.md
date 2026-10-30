# Personas and groups (section 7.3)

Nothing here asks a model for anything: it reads the SCD answers (the model rates a text as a person of each group compared in it) and the group positions of `kappa_by_group.csv`.

| file | content |
|---|---|
| `persona_group_matrix.csv` | per dataset, model, characteristic, persona (the group the model answered as) and group (the group whose position is compared): `n_texts`, `kappa`, `se`, `ci_low`, `ci_high`, `constant_label`. The cells with persona = group are the SCD kappas of `kappa_by_group.csv` |
| `persona_pairs.csv` | per dataset, model, characteristic and pair of personas: the kappa between their labels on the same texts, with the same columns |
| `persona_diagonal_summary.csv` | per dataset, characteristic and model (and `all` models): in how many (persona, group) comparisons the diagonal cell is higher than the off-diagonal cell of its column |

- A persona has a label only on the texts where its group is compared, so a cell uses the attributed texts where both its persona and its group are compared (`n_texts`; 0 gives no value).
- A constant label (the persona gives one label to every text of the cell, `constant_label = True`) gives kappa 0; in `persona_pairs.csv` the flag is set when either persona is constant.
- `se` and the 95% range (`ci_low`, `ci_high`) come from 10,000 resamples of the attributed texts of the characteristic (seed 42, the resamples of `kappa_by_group.csv`).
- `persona_diagonal_summary.csv` compares the point estimates (rounded to four decimals); `diagonal_higher`, `tie` and `diagonal_lower` count the pairs of one diagonal cell and one off-diagonal cell of the same column.
