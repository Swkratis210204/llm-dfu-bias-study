# Dataset notes

Facts about the four datasets for a reader, to sit next to the figures and tables in this folder. They describe all the texts after the preprocessing
(the cleaned ratings), with no AU and no DFU statistics. The file a number comes from is given in brackets (`eda/` is `new_results/<dataset>/eda/`).

## At a glance

| | popquorn | attitudes | MHS | Kumar |
|---|---|---|---|---|
| texts | 1,500 | 625 | 5,990 | 107,620 |
| ratings | 11,352 | 3,354 | 64,323 | 450,160 |
| ratings per text (min / median / max) | 3 / 8 / 10 | 2 / 6 / 11 | 2 / 4 / 777 | 1 / 4 / 5 |
| scale | 1-5 | 1-5 | 1-3 | 1-5 |
| mean rating (not comparable across scales) | 1.89 | 2.94 | 1.90 | 2.01 |
| ratings called toxic (rating 3 or more) | 26.4% | 54.0% | 42.4% | 29.7% |
| words per text, median (max) | 24 (99) | 20 (115) | 20 (118) | 15 (1,402) |
| annotators, raw / cleaned | 262 / 228 | not available | 7,912 / 7,577 | not available |

(Source: `eda/overview.csv`, `eda/text_length.csv`, `eda_overview_all.csv`.)

## The four notes

**MHS with and without the 70 reference comments.** Comments 20001-20070 are each rated by hundreds of annotators (234 to 777 ratings after the
cleaning; every other comment has 2 to 6). They are 41,442 of the 64,323 cleaned ratings (64.4%), so every rating-level number for MHS is mostly
theirs:

| | texts | ratings | rating 1 / 2 / 3 | toxic (rating 3) |
|---|---|---|---|---|
| all | 5,990 | 64,323 | 52.4% / 5.2% / 42.4% | 42.4% |
| without the 70 | 5,920 | 22,881 | 66.1% / 6.8% / 27.1% | 27.1% |
| the 70 only | 70 | 41,442 | 44.9% / 4.3% / 50.8% | 50.8% |

The total figures (`rating_distribution.png`, `toxicity.png`) show MHS with the 70 included. Without them it looks much more like popquorn and Kumar:
about two thirds of the ratings are the lowest value and a little over a quarter are toxic.

**MHS is nearly binary.** Only 5.2% of its ratings are "Unclear" (2); 52.4% are "No" and 42.4% are "Yes". Only "Yes" counts as toxic, so the 1-3
scale behaves like a yes/no judgment. It is also not on the 1-5 scale the models use. (Source: `rating_distribution.csv`.)

**Gen z is almost absent in Attitudes.** It has 4 of the 3,354 cleaned ratings (0.1%); millenial has 2,541 (75.8%) and gen x+ 809 (24.1%). It is
also small elsewhere: 8.6% of the ratings in MHS, 10.3% in Kumar and 11.8% in popquorn. (Source: `eda/groups.csv`.)

**Kumar texts that drop below four ratings.** Every Kumar text has 5 ratings in the raw data. The cleaning drops a rating when its rater has an
excluded value in any of the five dimensions (race alone removes 10.4% of the ratings), so afterwards 44,580 texts keep 5 ratings, 42,380 keep 4,
16,720 keep 3, 3,640 keep 2 and 300 keep 1: 20,660 texts (19.2%) have fewer than four. The same happens on a smaller scale elsewhere: 919 MHS texts
(15.3%), 41 Attitudes texts (6.6%) and 2 popquorn texts (0.1%). (Source: `ratings_per_text.csv`.)

## More facts

- **Where the texts come from.** Kumar records the source of each comment: Twitter 72,576 (67.4%), 4chan 18,854 (17.5%), Reddit 16,190 (15.0%). The
  other three files do not record a source for the texts. (Source: the raw Kumar file.)
- **Texts are short.** The median text has 15 to 24 words. Kumar has the shortest (median 15) and also the longest (1,402 words); the longest in the
  other three is 99 to 118. (Source: `eda/text_length.csv`.)
- **Ratings are skewed to the lowest value, except in Attitudes.** The lowest rating is 56.3% of the ratings in popquorn, 52.4% in MHS and 51.7% in
  Kumar. Attitudes is U-shaped: 31.1% rate 1 and 29.2% rate 5, with the middle values the least common. The share of toxic ratings therefore ranges
  from 26.4% to 54.0%. (Source: `rating_distribution.csv`, `toxicity.csv`.)
- **The scales are different.** Popquorn (`offensiveness`) and Attitudes (`toyou`) are 1-5 as collected; Kumar's `toxic_score` is 0-4 and MHS's
  `hatespeech` is 0-2, both shifted by one. A rating of 3 or more is toxic in all four, which for MHS means "Yes" only.
- **Raters are unbalanced across groups** (shares of the ratings):

  | | popquorn | attitudes | MHS | Kumar |
  |---|---|---|---|---|
  | gender | man 48.7 / woman 51.3 | man 56.4 / woman 43.6 | male 43.3 / female 56.7 | Male 47.8 / Female 52.2 |
  | race: White / Black / Asian | 78.6 / 13.5 / 7.9 | 66.3 / 33.7 / - | - | 79.9 / 13.7 / 6.4 |
  | age: gen z / millenial / gen x+ | 11.8 / 41.2 / 46.9 | 0.1 / 75.8 / 24.1 | 8.6 / 66.7 / 24.7 | 10.3 / 64.9 / 24.7 |
  | politics: left / center / right | - | 51.9 / 13.7 / 34.4 | 55.7 / 17.8 / 26.5 (liberal / neutral / conservative) | 42.8 / 27.8 / 29.4 (liberal / independent / conservative) |

  Popquorn also has education (college 48.8%, graduate 21.0%, high school 30.3%) and occupation (employed 51.3%, self-employed 14.0%, retired 11.9%,
  unemployed 11.0%, student 7.5%, homemaker 4.4%); Kumar has religion (high importance 32.4%, not high 67.6%); Attitudes has traditionalism (high 62.3%,
  not high 37.7%). (Source: `eda/groups.csv`, `toxicity_by_group.csv`.)
- **Group differences in the share of toxic ratings are small, except for race and religion.** Gender: 25.3 to 27.6% in popquorn, 27.9 to 31.7% in Kumar,
  41.9 to 42.7% in MHS, 51.4 to 57.3% in Attitudes. Black raters call more ratings toxic than White raters: 35.8% against 24.8% in popquorn (11.0 points),
  44.5% against 27.4% in Kumar (17.1) and 59.3% against 51.2% in Attitudes (8.1). In Kumar, raters who find religion very important: 39.4% against 25.0%;
  conservatives 35.3% against liberals 28.0%. In MHS the political and age groups are within 1 to 6 points. (Source: `toxicity_by_group.csv`.)
- **Annotator ids exist for two datasets.** Popquorn: 262 annotators, 228 after the cleaning, each with 45 to 50 ratings. MHS: 7,912 annotators, 7,577
  after the cleaning, with 1 to 21 ratings each (median 8). In both, no annotator has more than one value of an attribute and no (annotator, text) pair
  is repeated. Attitudes and Kumar have no annotator id, so annotator counts are not available. Kumar's comments come in batches of 20 that are all rated
  by the same 5 rater profiles, which is not enough to count people. (Source: `eda/annotators.csv`.)
- **Some rater attributes are missing or undisclosed.** Attitudes: 41 ratings without an age (1.2% of the raw ratings). MHS: age missing 105, ideology
  missing 27, "prefer not to say" for gender 500 (0.4%). Popquorn: occupation "prefer not to disclose" 50 (0.4%). Kumar: race missing 1,560, "prefer not to
  say" for politics 22,100 (4.1%), religion 7,400 (1.4%), age 1,140 (0.2%) and race 6,020 on its own. (Source: `eda/group_mapping.csv`.)
- **Kumar's race question allowed several answers.** The raw column has 58 distinct values plus missing; only White, Black or African American and Asian
  on their own are used. Hispanic raters (14,880 ratings, 2.8% of the raw ratings) and every multi-select answer are excluded, so they are absent from the
  race groups. Raters of other races are excluded in the other datasets too (popquorn: Hispanic or Latino, Native American, Arab American; Attitudes:
  hisp, native, other, middleEastern). (Source: `eda/group_mapping.csv`.)
- **The preprocessing keeps different shares of the data.** Ratings kept: popquorn 87.1%, Attitudes 96.9%, Kumar 83.7%, MHS 47.5%. MHS also keeps only
  5,990 of 39,565 comments (15.1%), because a comment needs at least four ratings to be kept, and most MHS comments have one to three. (Source:
  `eda/preprocessing.csv`.)
- **Attitudes, ratings per tweet.** In the raw file 450 of the 626 tweets have 6 ratings, 122 have 4, 52 have 5, one has 3 and one has 12. One tweet is
  lost in the cleaning (625 left). (Source: the raw Attitudes file.)
- **Group names are written differently in each dataset** (Man, man, male, Male; White, white; ...). The combined table `toxicity_by_group.csv` harmonizes
  them, and pools the three political measures (Attitudes `politics`, Kumar `politics`, MHS `ideology`) even though they are not the same question.
