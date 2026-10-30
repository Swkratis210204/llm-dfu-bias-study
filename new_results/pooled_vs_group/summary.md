# Pooled kappa vs. group kappa (default prompt)

Cohen's kappa of the model's default-prompt label against the pooled label of the attributed texts and against each group's position. Bootstrap: 10,000 resamples of the texts, seed 42 (those of the existing tables). † = the model gave one label to every text (kappa set to 0). Group kappas are identical to Table 4 and Tables 16-18 (checked by the script). Kumar: the 65-attribution sample.

## 1. Kappa by dataset and characteristic

### attitudes

**gender**

| model | pooled (n=10) | man (n=10) | woman (n=10) |
|---|---|---|---|
| ChatGPT | 0.21 | -0.40 | 0.40 |
| DeepSeek | 0.00† | 0.00† | 0.00† |
| Gemini | 0.00† | 0.00† | 0.00† |
| Mistral | 0.41 | 0.20 | -0.20 |

**race**

| model | pooled (n=14) | black (n=14) | white (n=14) |
|---|---|---|---|
| ChatGPT | 0.38 | -0.26 | 0.21 |
| DeepSeek | 0.51 | -0.24 | 0.11 |
| Gemini | 0.66 | -0.21 | 0.26 |
| Mistral | 0.10 | 0.42 | -0.07 |

**age**

| model | pooled (n=6) | gen x+ (n=6) | millenial (n=6) |
|---|---|---|---|
| ChatGPT | -0.20 | -0.80 | 0.25 |
| DeepSeek | -0.20 | -0.20 | 0.25 |
| Gemini | 0.40 | -0.20 | 1.00 |
| Mistral | 0.18 | -0.36 | 0.57 |

**politics**

| model | pooled (n=12) | left (n=12) | right (n=12) |
|---|---|---|---|
| ChatGPT | 0.56 | 0.40 | 0.00 |
| DeepSeek | 0.56 | 0.40 | 0.00 |
| Gemini | 0.43 | 0.31 | 0.31 |
| Mistral | 0.25 | 0.57 | -0.29 |

**traditionalism**

| model | pooled (n=9) | high (n=9) | not_high (n=9) |
|---|---|---|---|
| ChatGPT | 0.18 | 0.18 | -0.42 |
| DeepSeek | 0.18 | 0.18 | -0.42 |
| Gemini | -0.20 | -0.20 | 0.27 |
| Mistral | -0.20 | -0.20 | 0.27 |

### popquorn

**gender**

| model | pooled (n=14) | Man (n=14) | Woman (n=14) |
|---|---|---|---|
| ChatGPT | 0.59 | 0.04 | 0.16 |
| DeepSeek | 0.43 | -0.14 | 0.29 |
| Gemini | 0.43 | -0.14 | 0.29 |
| Mistral | 0.02 | -0.23 | 0.34 |

**race**

| model | pooled (n=13) | Black or African American (n=13) | White (n=13) |
|---|---|---|---|
| ChatGPT | 0.03 | 0.27 | -0.15 |
| DeepSeek | 0.16 | 0.15 | -0.14 |
| Gemini | 0.15 | 0.16 | 0.07 |
| Mistral | 0.15 | 0.16 | 0.07 |

**age**

| model | pooled (n=55) | gen x+ (n=55) | millenial (n=55) |
|---|---|---|---|
| ChatGPT | 0.19 | 0.12 | 0.19 |
| DeepSeek | 0.24 | 0.20 | 0.07 |
| Gemini | 0.20 | 0.15 | 0.22 |
| Mistral | 0.06 | 0.17 | -0.02 |

**education**

| model | pooled (n=47) | College degree (n=44) | Graduate degree (n=17) | High school diploma or equivalent (n=38) |
|---|---|---|---|---|
| ChatGPT | -0.02 | 0.22 | 0.07 | 0.20 |
| DeepSeek | 0.01 | 0.17 | 0.33 | 0.13 |
| Gemini | 0.22 | 0.28 | -0.34 | 0.29 |
| Mistral | 0.05 | 0.28 | -0.07 | 0.13 |

**occupation**

| model | pooled (n=54) | Employed (n=52) | Retired (n=11) | Self-employed (n=28) | Student (n=11) | Unemployed (n=10) |
|---|---|---|---|---|---|---|
| ChatGPT | 0.22 | 0.30 | 1.00 | 0.19 | -0.22 | 0.09 |
| DeepSeek | 0.20 | 0.41 | 0.79 | 0.21 | -0.52 | 0.29 |
| Gemini | 0.23 | 0.48 | 1.00 | 0.16 | -0.52 | 0.14 |
| Mistral | 0.15 | 0.26 | 0.35 | 0.12 | 0.03 | 0.02 |

### mhs

**gender**

| model | pooled (n=43) | female (n=43) | male (n=43) |
|---|---|---|---|
| ChatGPT | 0.01 | 0.07 | -0.03 |
| DeepSeek | 0.00 | 0.06 | -0.03 |
| Gemini | 0.00 | 0.20 | -0.10 |
| Mistral | 0.00 | 0.06 | -0.03 |

**age**

| model | pooled (n=30) | gen x+ (n=30) | millenial (n=30) |
|---|---|---|---|
| ChatGPT | 0.00 | -0.12 | 0.05 |
| DeepSeek | 0.00 | -0.06 | 0.02 |
| Gemini | 0.00† | 0.00† | 0.00† |
| Mistral | 0.00 | -0.06 | 0.02 |

**ideology**

| model | pooled (n=32) | conservative (n=22) | liberal (n=29) | neutral (n=13) |
|---|---|---|---|---|
| ChatGPT | 0.00 | -0.09 | 0.10 | 0.05 |
| DeepSeek | 0.00† | 0.00† | 0.00† | 0.00† |
| Gemini | 0.00 | 0.00† | 0.05 | 0.05 |
| Mistral | 0.00 | -0.09 | 0.05 | 0.00† |

### kumar

**gender**

| model | pooled (n=10) | Female (n=10) | Male (n=10) |
|---|---|---|---|
| ChatGPT | -0.25 | 0.40 | -0.40 |
| DeepSeek | -0.25 | 0.40 | -0.40 |
| Gemini | 0.00 | 1.00 | -1.00 |
| Mistral | -0.40 | 0.60 | -0.60 |

**race**

| model | pooled (n=15) | Asian (n=10) | Black (n=10) | White (n=10) |
|---|---|---|---|---|
| ChatGPT | 0.36 | 0.60 | 0.41 | -0.36 |
| DeepSeek | 0.44 | 0.60 | 0.29 | -0.32 |
| Gemini | -0.06 | 0.20 | 0.20 | -0.36 |
| Mistral | 0.22 | 0.20 | -0.15 | 0.00 |

**age**

| model | pooled (n=15) | gen x+ (n=10) | gen z (n=10) | millenial (n=10) |
|---|---|---|---|---|
| ChatGPT | 0.31 | -0.32 | 0.60 | -0.30 |
| DeepSeek | 0.44 | 0.00† | 0.20 | 0.05 |
| Gemini | 0.23 | 0.29 | 0.20 | 0.09 |
| Mistral | 0.39 | 0.00 | 0.80 | -0.03 |

**politics**

| model | pooled (n=15) | conservative (n=10) | independent (n=10) | liberal (n=10) |
|---|---|---|---|---|
| ChatGPT | 0.47 | -0.09 | -0.07 | 1.00 |
| DeepSeek | 0.17 | 0.07 | -0.40 | 0.78 |
| Gemini | 0.47 | -0.09 | -0.07 | 1.00 |
| Mistral | 0.36 | 0.05 | -0.07 | 0.80 |

**religion**

| model | pooled (n=10) | high (n=10) | not_high (n=10) |
|---|---|---|---|
| ChatGPT | 0.23 | -0.19 | 0.29 |
| DeepSeek | 0.23 | -0.19 | 0.29 |
| Gemini | 0.00 | -0.20 | 0.40 |
| Mistral | 0.44 | -0.18 | 0.19 |

## 2. Kumar gender

| model | pooled | female | male |
|---|---|---|---|
| ChatGPT | -0.25 | 0.40 | -0.40 |
| DeepSeek | -0.25 | 0.40 | -0.40 |
| Gemini | 0.00 | 1.00 | -1.00 |
| Mistral | -0.40 | 0.60 | -0.60 |

Pooled kappa with its bootstrap SD and 2.5-97.5 percentile range:

| model | pooled kappa | SD | 2.5% | 97.5% |
|---|---|---|---|---|
| ChatGPT | -0.25 | 0.28 | -0.67 | 0.35 |
| DeepSeek | -0.25 | 0.28 | -0.67 | 0.35 |
| Gemini | 0.00 | 0.30 | -0.60 | 0.60 |
| Mistral | -0.40 | 0.27 | -0.80 | 0.20 |

## 3. Counts over all datasets (model x characteristic cases; every characteristic has at least two groups)

- cases: 72
- pooled kappa strictly between the lowest and highest group kappa: 45
- |pooled| < 0.2 while the group kappas differ by >= 0.5: 8

| dataset | cases | pooled strictly between | |pooled| < 0.2 and spread >= 0.5 |
|---|---|---|---|
| attitudes | 20 | 6 | 3 |
| popquorn | 20 | 13 | 1 |
| mhs | 12 | 9 | 0 |
| kumar | 20 | 17 | 4 |

## 4. Caveats and checks

- Pooled label: all ratings of the text in the cleaned data (all annotators, also those of groups removed by the rule of ten AU texts; ratings with a removed value in any characteristic were dropped by the preprocessing, so the same pooled label is used for every characteristic of a text). Toxic if most ratings are >= 3 (MHS: only 3), an exact tie decided by the mean (>= 3 toxic).

Pooled ties resolved by the mean, per dataset (distinct attributed texts / attributed rows = text x characteristic):

| dataset | distinct texts | ties (texts) | rows | ties (rows) |
|---|---|---|---|---|
| attitudes | 41 | 11 | 51 | 14 |
| popquorn | 154 | 21 | 183 | 28 |
| mhs | 95 | 91 | 105 | 100 |
| kumar | 65 | 15 | 65 | 15 |

Per characteristic:

| dataset | characteristic | attributed texts | ties by the mean | pooled toxic | pooled civil |
|---|---|---|---|---|---|
| attitudes | gender | 10 | 5 | 7 | 3 |
| attitudes | race | 14 | 4 | 9 | 5 |
| attitudes | age | 6 | 2 | 2 | 4 |
| attitudes | politics | 12 | 1 | 9 | 3 |
| attitudes | traditionalism | 9 | 2 | 6 | 3 |
| popquorn | gender | 14 | 5 | 5 | 9 |
| popquorn | race | 13 | 5 | 2 | 11 |
| popquorn | age | 55 | 6 | 5 | 50 |
| popquorn | education | 47 | 7 | 6 | 41 |
| popquorn | occupation | 54 | 5 | 7 | 47 |
| mhs | gender | 43 | 40 | 1 | 42 |
| mhs | age | 30 | 30 | 0 | 30 |
| mhs | ideology | 32 | 30 | 0 | 32 |
| kumar | gender | 10 | 5 | 4 | 6 |
| kumar | race | 15 | 4 | 3 | 12 |
| kumar | age | 15 | 1 | 4 | 11 |
| kumar | politics | 15 | 2 | 6 | 9 |
| kumar | religion | 10 | 3 | 4 | 6 |

- Pooled rows with a constant model label (kappa 0, †): 4 of 72.
- Group rows with a constant model label: 12 of 176.
- Cases where the pooled label is the same for every attributed text of the characteristic (pooled kappa is then 0 by construction): mhs/age, mhs/ideology.
- Datasets where most pooled labels are decided by the mean (exact tie between toxic and civil ratings): mhs (91 of 95 texts). There the pooled label is mostly a mean-threshold decision.
- Share of texts where the pooled label equals the group position: column `share_pooled_equals_group` of the csv.
- Match with the paper: all group rows (kappa, n, SE, interval, constant flag) are identical to `new_results/kappa/default/kappa_by_group.csv` (Tables 4, 16-18).
