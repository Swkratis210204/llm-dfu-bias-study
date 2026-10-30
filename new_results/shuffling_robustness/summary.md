# Permutation null for the attribution rates

Chance = the AU test rerun after shuffling, on every polarized text with at least four ratings, the group labels among the ratings of the retained groups of the characteristic (group sizes and the text's ratings unchanged; thin groups not removed again; every characteristic shuffled independently, seed 42). p = (1 + permutations with attributed >= observed) / (1 + permutations). Ranges are 2.5–97.5 percentiles. D = demographic (gender, race, age), N = non-demographic. The observed testable counts, attributed counts and rates, the pooled rates and the gap with its range reproduce Tables 2 and 3 (checked against `attribution_gap`; the testable texts of every permutation equal the observed ones, asserted).

## Attitudes (1000 permutations)

| characteristic | type | n groups | testable | observed attributed | observed % | chance % | chance range | excess (pp) | ratio | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | D | 2 | 164 | 10 | 6.1 | 4.0 | [1.2, 7.3] | +2.1 | 1.51 | 0.1119 |
| race | D | 2 | 239 | 14 | 5.9 | 4.8 | [2.5, 7.5] | +1.0 | 1.21 | 0.2617 |
| age | D | 3 | 97 | 6 | 6.2 | 4.2 | [1.0, 8.2] | +1.9 | 1.46 | 0.2268 |
| politics | N | 2 | 144 | 12 | 8.3 | 7.0 | [3.5, 10.4] | +1.3 | 1.19 | 0.2787 |
| traditionalism | N | 2 | 154 | 9 | 5.8 | 3.8 | [1.3, 7.1] | +2.0 | 1.53 | 0.1139 |

## POPQUORN (1000 permutations)

| characteristic | type | n groups | testable | observed attributed | observed % | chance % | chance range | excess (pp) | ratio | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | D | 2 | 615 | 14 | 2.3 | 2.6 | [1.5, 3.9] | -0.3 | 0.87 | 0.7542 |
| race | D | 2 | 204 | 13 | 6.4 | 4.5 | [2.0, 7.3] | +1.9 | 1.41 | 0.1289 |
| age | D | 2 | 583 | 55 | 9.4 | 8.5 | [7.0, 9.9] | +1.0 | 1.12 | 0.1299 |
| education | N | 3 | 591 | 47 | 8.0 | 6.4 | [4.6, 8.3] | +1.6 | 1.25 | 0.0589 |
| occupation | N | 5 | 418 | 54 | 12.9 | 12.6 | [9.8, 15.6] | +0.3 | 1.02 | 0.4486 |

## MHS (1000 permutations)

| characteristic | type | n groups | testable | observed attributed | observed % | chance % | chance range | excess (pp) | ratio | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | D | 2 | 529 | 43 | 8.1 | 8.4 | [6.4, 10.6] | -0.2 | 0.97 | 0.6244 |
| age | D | 2 | 305 | 30 | 9.8 | 7.8 | [5.2, 10.5] | +2.0 | 1.26 | 0.0809 |
| ideology | N | 3 | 336 | 32 | 9.5 | 8.9 | [6.2, 11.6] | +0.6 | 1.07 | 0.3656 |

## Kumar (1000 permutations)

| characteristic | type | n groups | testable | observed attributed | observed % | chance % | chance range | excess (pp) | ratio | p |
|---|---|---|---|---|---|---|---|---|---|---|
| gender | D | 2 | 19436 | 1224 | 6.3 | 5.7 | [5.4, 6.0] | +0.6 | 1.10 | 0.0010 |
| race | D | 3 | 5727 | 667 | 11.7 | 8.0 | [7.3, 8.6] | +3.7 | 1.46 | 0.0010 |
| age | D | 3 | 11400 | 1038 | 9.1 | 8.7 | [8.2, 9.2] | +0.4 | 1.04 | 0.0719 |
| politics | N | 3 | 15584 | 1924 | 12.3 | 11.8 | [11.4, 12.3] | +0.5 | 1.04 | 0.0150 |
| religion | N | 2 | 15940 | 1237 | 7.8 | 5.8 | [5.4, 6.1] | +2.0 | 1.35 | 0.0010 |

## The gap (non-demographic minus demographic, percentage points)

Bootstrap: 10,000 resamples of the polarized texts (seed 42, as Table 3). In every resample the observed rates are recomputed and the chance rates are the mean of the per-pair chance probabilities of the resampled pairs.

| dataset | observed D % | observed N % | observed gap [range] | share <= 0 | chance D % | chance N % | excess gap [range] | share <= 0 |
|---|---|---|---|---|---|---|---|---|
| Attitudes | 6.0 | 7.0 | +1.1 [-2.4, +4.6] | 28.2% | 4.5 | 5.3 | +0.2 [-3.0, +3.5] | 46.4% |
| POPQUORN | 5.8 | 10.0 | +4.2 [+2.1, +6.3] | 0.0% | 5.3 | 9.0 | +0.5 [-1.4, +2.4] | 29.5% |
| MHS | 8.8 | 9.5 | +0.8 [-2.7, +4.3] | 34.1% | 8.2 | 8.9 | +0.0 [-3.0, +3.1] | 49.9% |
| Kumar | 8.0 | 10.0 | +2.0 [+1.6, +2.4] | 0.0% | 7.0 | 8.8 | +0.3 [-0.1, +0.7] | 9.9% |

## Spearman correlation over all characteristics and datasets

| measure | Spearman | characteristics |
|---|---|---|
| n_groups vs observed % | +0.572 | 18 |
| n_groups vs chance % | +0.572 | 18 |

## Permutations used

| dataset | permutations |
|---|---|
| attitudes | 1000 |
| popquorn | 1000 |
| mhs | 1000 |
| kumar | 1000 |

## Caveats

- Chance is computed with the groups retained by the paper's 10-text removal rule; the rule is not rerun inside the permutations.
- A permutation keeps every group's size on every text, so a text with a group of one rating is never made testable or untestable; only which ratings fall in which group changes.
- Characteristics with fewer than 15 testable texts (their chance range and p are coarse): none.
- The per-pair chance probabilities are in `per_pair_chance_probability.csv` (one row per testable text and characteristic).
