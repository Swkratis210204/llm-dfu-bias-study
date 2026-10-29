# AU texts across the datasets (exploratory)

A text is **polarized** when its ratings are not unimodal (DFU above 0). It is **AU** (a-posteriori unimodal) for a characteristic when it is polarized, at
least two annotator groups of that characteristic have two or more ratings on it, and every one of those groups is internally unimodal. The AU texts of a
characteristic are then the texts on which the disagreement is explained by who the annotators are. A text can be AU for several characteristics; a **row**
is one text under one characteristic. A group in fewer than 10 AU texts (counted per characteristic) is removed and the analysis is repeated, unless every
group is below 10 or only two groups are left (then nothing is removed). All numbers use the cleaned ratings.

## Files

| file | content |
|---|---|
| `au_summary.csv` | per dataset: AU texts and rows, the dimensions a text is AU in, ratings, DFU and toxic share of the AU texts, rows given to the models |
| `au_funnel.csv`, `au_funnel.png` | what happens to the texts of every dimension: not polarized, too few groups, a group not unimodal, AU |
| `au_groups.csv`, `au_texts_per_group.png` | per group: the AU texts it takes part in, its status under the rule of 10, its ratings and toxic share in the AU texts |
| `au_label_patterns.csv`, `au_label_patterns.png` | AU texts whose groups give different labels, all toxic, all civil; the group labels decided by the mean |
| `au_dimension_overlap.csv`, `au_dimension_overlap.png` | how many AU texts are AU in more than one dimension |
| `au_rating_distribution.csv`, `au_rating_distribution.png/.pdf` | the ratings of the AU texts (toxic ratings in red) |
| `au_toxicity_by_dimension.png/.pdf` | the share of toxic ratings per group in the AU texts, the datasets side by side (groups removed under the rule are left out) |
| `au_models_input.csv`, `au_kumar_sample.png` | each group's share of all AU rows against the rows given to the models (they differ only for Kumar) |

## Be careful

- **The AU set is small**, and for some groups very small: Attitudes has 11 to 41 AU ratings per group. Read small gaps in the AU toxic shares with care.
- **"Kept, below 10"** (orange in `au_texts_per_group.png`) are groups that the rule could not remove: all groups of the characteristic were below 10 (Attitudes
  age) or only two groups were left (Attitudes traditionalism). They are in the analysis with fewer than 10 AU texts.
- **Kumar:** the AU set has 6,090 rows; the models were given a 65-row sample (every group at least 10 times, not proportional). `au_models_input.csv`
  and `au_kumar_sample.png` show how the sample differs.
- **Group labels** are the majority of the group's binarized ratings; an exact tie is decided by the group's mean rating (`decided by the mean`).
- **AU depends on the number of ratings:** a text needs at least four ratings, and with few ratings DFU takes few values (see `new_results/total_dfu/`).
