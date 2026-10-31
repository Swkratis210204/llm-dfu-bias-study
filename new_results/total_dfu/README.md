# DFU across the datasets (exploratory)

DFU is the normalized distance from unimodality of the histogram of a text's ratings (the `ndfu` package): 0 means the ratings are unimodal, larger
values mean the ratings split into separate peaks (1 is the largest). A text is **polarized** when its DFU is above the tolerance (0 for all four
datasets, so any DFU above 0), and **AU** when, in addition, at least two annotator groups have two or more ratings on it and every one of those
groups is unimodal. All numbers use the cleaned ratings (after the preprocessing); the DFU values are the ones the AU analysis computed.

## Files

| file | content |
|---|---|
| `dfu_text_level.csv` | one row per dataset and text: ratings, DFU, polarized, AU (in any dimension), number of AU dimensions |
| `dfu_summary.csv` | per dataset: texts, polarized share, statistics of the DFU of all texts, polarized texts and AU texts |
| `dfu_histogram.csv` | the DFU in bins of 0.05 (all, polarized, AU): the numbers behind the histograms |
| `dfu_by_ratings_per_text.csv` | share of polarized texts and mean DFU by the number of ratings of a text |
| `dfu_group_level.csv` | per dimension and group: how often the group's own ratings on a text are unimodal (the second condition of AU) |
| `dfu_pairwise_ks.csv` | Kolmogorov-Smirnov distance between the DFU distributions of every two datasets |
| `dfu_funnel.csv` | the funnel of every dimension (not polarized, too few groups, a group not unimodal, AU), merged |
| `dfu_all_texts.png` | the DFU of all texts, one panel with a bar color per dataset (the spike at 0 is the texts that are not polarized) |
| `dfu_polarized_texts.png` | the DFU of the polarized texts, with the AU texts among them |
| `dfu_ecdf.png` | cumulative share of texts by DFU, all datasets in one panel (all texts, and polarized texts) |
| `dfu_by_ratings_per_text.png` | share of polarized texts and mean DFU by the number of ratings of a text |
| `dfu_boxplot_all_texts.png` | the DFU of all texts, one box per dataset (the median is 0 in every dataset: more than half of the texts are not polarized) |
| `dfu_boxplot_au_texts.png` | the DFU of the AU texts, one box per dataset |

## Be careful when comparing datasets

- **The DFU depends on how many ratings a text has.** With few ratings only a few DFU values are possible (see `distinct DFU values` in
  `dfu_summary.csv`), and a text with few ratings is more often unimodal by chance. Kumar texts have at most 5 ratings; popquorn up to 10; MHS up to 777
  (70 reference comments, see the dataset notes); `dfu_by_ratings_per_text` shows the effect. In `dfu_by_ratings_per_text.png` a point is drawn only
  when it rests on at least 20 texts; the csv has all of them.
- **The scales differ.** MHS is 1-3, the others 1-5, so the same DFU does not mean the same disagreement.
- **A group is not in `dfu_group_level.csv`** when it was removed from the AU analysis (fewer than 10 AU texts).
- **The Kolmogorov-Smirnov distance is descriptive:** with thousands of texts a p-value is meaningless, and the datasets differ in more than the DFU.
