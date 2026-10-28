# Notes for the manuscript: irregularities, choices and limits to report

Things found while running the experiments that a reader or reviewer should be told. Each entry: what, the numbers, where it comes from, where to
mention it, and whether the decision is open. Add to this file whenever something unusual turns up. Last updated 2026-10-27.

## Data and preprocessing

**1. MHS: the prefilter removes most of the data (decided: report).**
MHS comments need at least 4 ratings to be able to be AU (2 + 2), so the prefilter drops the rest: 39,565 comments become 5,990 (85% lost) and
135,556 ratings become 64,323 (47.5% kept). Most MHS comments have only 1-3 ratings. Source: `new_results/mhs/eda/preprocessing.csv`.
Where: dataset description / limitations.

**2. MHS: 70 "anchor" comments carry 64% of the ratings (OPEN: how to report).**
Comments 20001-20070 are each rated by about 800 different annotators (the largest has 815; no repeated annotator, no duplicated text);
every other comment has 2-6 ratings in the cleaned data. They are 41,442 of the 64,323 clean ratings (64.4%). None of them is AU (AU texts have 4-5 ratings), so the AU results
do not change, but every rating-level number for "all texts" is dominated by them: share of toxic ratings 42.4% with them, 27.1% without; group
rates move the same way (female 42.7 -> 27.6, male 41.9 -> 26.5). Flagged automatically in `overview` ("texts with > 3x the median ratings").
Decision still open: report both numbers, or leave the anchors out of the "all texts" column. Source: `new_results/mhs/eda/overview.csv`.

**3. Ratings removed for an excluded group value.**
Groups with too few raters were excluded before the AU analysis, and a rating with an excluded value in any dimension is dropped. Share of the raw
ratings lost this way: popquorn 12.9%, attitudes 3.1%, mhs 2.1%, Kumar 16.3% (Kumar race alone: 10.4%). Lists of the excluded values are in the
dataset sections of the notebook. Source: each `eda/preprocessing.csv`. Where: preprocessing paragraph.
Per-dimension counts there are taken after the prefilter but the percentage column divides by the raw total (understates them for MHS).

**4. Groups removed from the AU analysis (thin groups).**
A group in fewer than 10 AU texts is removed. popquorn: Asian (race), gen z (age), Homemaker (occupation). MHS: gen z (age; 0 AU ratings). Source:
`eda/groups.csv`, column `kept in AU analysis`.

**5. Different rating scales.**
MHS is 1-3 (No / Unclear / Yes; only Yes counts as toxic); the others are 1-5. Toxic is a rating of 3 or more everywhere. The models always use 1-5
and the rule 1-2 civil, 3-5 toxic, so MHS human ratings and model scores are not on the same scale. Needs a sentence wherever they are compared.

**6. Kumar: the models saw a sample, not the AU set.**
Kumar has 5,209 AU texts (6,090 text-dimension rows); the models annotated 65 rows, the smallest sample in which every group of every dimension
appears at least 10 times (not proportional on purpose). The EDA describes the full AU set; every model analysis concerns the 65. Source:
`new_results/kumar/kumar_au_sample.csv`. Where: Kumar section, limitations. A larger sample was considered and parked (it would break the
protocol shared by the four datasets); possible as a supplementary robustness check.

**7. Attitudes: texts and rows.**
Attitudes has 51 text-dimension rows but 41 distinct texts (a text can be AU in several dimensions); the baseline is per distinct text, the SCD run per
row. One text was lost in the cleaning (626 -> 625). The prompt-sensitivity sample for attitudes is all 41 texts, not 50.

**8. Most AU texts are not AU in every dimension, and in some datasets the groups often do not disagree.**
Share of AU rows whose groups have different labels: popquorn 87 of 183 (gender 79%, race 92%, age 40%, education 53%, occupation 31%); MHS 93-100%;
Kumar 59-97% (gender 96%, race 87%, age 70%, politics 59%, religion 97%). Where the groups agree, "matched" and "mismatched" persona comparisons
cannot differ; the human-versus-SCD comparison is informative on the disagreeing rows. Source: each `eda/au_labels.csv`.

## Models and prompts

**9. Four of five planned models were run; Llama was not.**
ChatGPT (gpt-4o-mini), DeepSeek (deepseek-chat), Mistral (open-mistral-7b) and Gemini (gemini-2.5-flash) were run on everything. Llama (llama3:8b) was
set aside: it runs locally on a 4 GB GPU (partly on the CPU) and is too slow. Exact model ids and settings: `new_results/run_log.csv`. Hosted models
can change behind the same name, and temperature 0 does not guarantee identical answers on another day; runs were done on 2026-10-27.

**10. Prompt development.**
The baseline prompt states the rule "1 or 2 is civil, 3, 4 or 5 is toxic". An earlier round without the rule (84 of 1,420 model labels contradicted the
score) was discarded; the label is now always derived from the score. The prompts were settled on the same texts that are analysed (there is no held-out
set, for any of the four datasets); the prompt-sensitivity result (below) is the evidence that the exact wording matters little.
Exact wordings of all four prompts (baseline, two rewordings, SCD): `new_results/prompts.md`.

**11. Prompt sensitivity (6.2): what it does and does not show.**
Sample of 50 random texts per dataset (41 for attitudes), two paraphrases of the baseline, one run each. Against the baseline: same label for 91-92% of
texts, weighted kappa about 0.80; the two paraphrases agree more with each other (kappa 0.88). Mistral is the least stable model; Gemini and Mistral
have low kappa on MHS (about 0.45) although 94-98% of their labels agree (cause not checked: probably a narrow score range). Label kappa is not
reported as a main result (it falls to 0 when almost every text has the same label). Only paraphrases that add no information were tested; a
definition, example or persona was not. Say "largely robust", not "independent". Source: Table 1 and 2 in notebook 6.2.6.

**12. SCD: the groups of a text are rated in one call (design limit).**
For a text with two groups the model answers once for both, so the ratings can influence each other. In a ChatGPT pilot of 20 rows, 14 got identical
scores for both groups. The order of the groups is a seeded shuffle, saved with each answer. A single-persona run on a subset would measure the effect;
not done yet. Where: methods and limitations.

**13. SCD: format notes.**
All 16 files (4 datasets x 4 models) are complete: 404 calls per model, all answered on the first attempt, no incomplete replies, every label follows
its score, every group matched to its human label and to the baseline. Mistral and Gemini wrap the JSON in code fences (handled by the parser);
for 6 of 375 Mistral groups the reply had no readable written label (not inspected one by one), so the label comes from the score. Checked 2026-10-27.

## Not available / not done yet

**14. Human scores are not in the AU file.** `aposteriori_for_models.csv` has the group label (majority, ties decided by the mean) but not the group's
mean rating. Comparing model scores with human scores needs those means joined from the raw datasets. Labels only until then.

**15. Group names are written as in each dataset.** The same group is "Man", "male" or "Male" depending on the dataset; the persona prompts use the
dataset's own string, so joins are exact but pooled tables across datasets need a mapping.

## Annotators and group definitions (added 2026-10-28)

Source for this whole section: `new_results/<dataset>/eda/annotators.csv`, `groups_annotators.csv`, `group_mapping.csv` and `new_results/eda_overview_all.csv`.

**16. Only popquorn and MHS have an annotator id.**
Attitudes and Kumar have none (the raters are known only by the attributes attached to each rating), so annotator counts and ratings per annotator are
"not available" for them. Popquorn: 262 annotators in the raw data, 228 after the cleaning (34 lost); each remaining annotator has 45-50 ratings
(median 50). MHS: 7,912 annotators raw, 7,577 clean (335 lost); ratings per annotator after the cleaning: 1 / 8 / 21 (min / median / max).
Where: dataset descriptions; do not report annotator numbers for attitudes and Kumar.

**17. Checks that came out clean (popquorn and MHS).**
No annotator has more than one value of any attribute (gender, race, age, education, occupation; gender, age, ideology), and no (annotator, text)
pair is repeated. Not checkable for attitudes and Kumar (no annotator id).

**18. Kumar: race is multi-select, and most raw values are excluded.**
The raw `race` column has 58 distinct values, plus missing (a rater can tick several races). Only White, Black or African American and Asian alone are used; every
combination ("White,Asian", "White,Hispanic", ...) is excluded, and so is Hispanic on its own (14,880 ratings). Hispanic raters are therefore absent from
Kumar's race groups, as Hispanic or Latino and Native American are in popquorn and `hisp` / `native` in attitudes. Race has 56,120 excluded ratings
(10.4% of the raw ratings). Where: group definitions; limitation on who is represented.

**19. Group mapping choices a reader may question.**
- Kumar `Under 18` (120 ratings, minors) is mapped to gen z together with 18-24.
- Popquorn age bands are used as in the file, including the band spelled `54-59`; attitudes `age` is a number (18, 25, ..., 65) and 18 gives gen z with
  only 4 ratings (and 41 ratings have no age).
- MHS age is in years and is cut at 25 and 45 (gen z 18-24, millenial 25-44, gen x+ 45-81).
- Attitudes `politics` and `traditionalism` are scores cut into groups (politics: left up to -0.25, center above -0.25 up to 0.25, right above 0.25;
  traditionalism: not_high 1-3, high 4-5). Politics `center` has only 467 ratings and is removed from the AU analysis.
- The "reason" column of `group_mapping.csv` is a simple rule on the raw text ("missing", "prefer not to say", "other, unknown or no opinion",
  otherwise "not part of the group definition of the dataset"); the real reasons are the modelling choices described in the dataset sections.

**20. Spelling.** The group is spelled `millenial` in all the results (as in the code); plot labels use American spelling ("polarization"), but the csv
column names and some notebook text still say "polarised".

**21. `dataset_info.csv` has TODO fields.** Source of the texts, task and the meaning of each scale value cannot be read from the data and are to be
filled in by hand (or from the dataset papers).

## Annotator identifiers in Attitudes and Kumar (checked 2026-10-28)

Source: the raw files `Thesis/Code/attitudes_embedded.csv` (a pickled DataFrame) and `datasets/kumar/toxicity_ratings.json`, and the paper
`Related Work/sapattitudes.pdf` (Sap et al., "Annotators with Attitudes"). The project has no documentation of the Kumar data other than the one-line
`datasets/README.md`; the statement that Kumar's participants rated batches of 20 comments comes from the Kumar paper, which is not in the project.

**22. Attitudes: annotators identifiable: NO.**
The raw file has 16 columns and no id, worker or session column: `tweet` (626 distinct) and 15 list columns parallel to the 3,463 ratings (`toyou`,
`toany`, `intent`, `racism`, `racist`, `freeSpeech`, `harmHateSpeech`, `lingPurism`, `traditionalism`, `annotatorPolitics`, `annotatorRace`,
`annotatorAge`, `annotatorGender`, `annotatorMinority` (free text, 39 distinct strings), `bert` (one 768-number vector per tweet)). The paper says the study
data are anonymized ("please contact the authors", footnote 2) and describes the breadth-of-posts study as 173 annotators, 571 posts and 6 ratings per
post (two from white conservative, two from white liberal and two from Black participants). The file does not match that: 626 tweets, 3-12 ratings per
tweet (450 tweets with 6, 122 with 4, 52 with 5, one with 3 and one with 12). The number of distinct combinations of the annotator-level columns is
248 (10 columns), 182 (6 columns) or 156 (the 5 columns used by the AU analysis); none equals the 173 annotators of the paper, so these combinations are
not annotator ids and none was built. Annotator counts stay "not available". Open: where the 626 tweets come from, since the paper reports 571 posts.

**23. Kumar: no annotator id, but a batch structure is visible (NOT used as an id).**
The raw file has comment-level keys `comment`, `source` (3 values), `perspective_score`, `comment_id` (unique) and `ratings`, and 25 keys per rating:
`toxic_score` (0-4), five `is_*` flags, `fine_to_see_online`, `remove_from_online`, and 17 rater attributes (gender, race, age_range, education,
political_affilation, religion_important, lgbtq_status, is_parent, identify_as_transgender, technology_impact, toxic_comments_problem, four `uses_media_*`
and two `personally_*`). No id, worker, session or hash. The pipeline reads only 6 of the 25 (the rating and five attributes).
`comment_id` is `<batch>_<position>`: 5,381 batches of exactly 20 comments each, position values 1 and 5-23 (positions 2, 3 and 4 never occur; cause not
known). Every comment has 5 ratings, and in all 5,381 batches the 20 comments have the same 5 rater profiles (all 17 attributes) in the same order, with
5 distinct profiles per batch. So within a batch, rating position k was given by the same rater (or at least the same profile) for all 20 comments, which
fits "participants rated batches of 20 comments". That gives 5,381 x 5 = 26,905 rater slots raw (538,100 ratings / 20) and 22,508 after the cleaning
(450,160 / 20; a slot is kept or removed as a whole because the attributes do not change inside a batch). These are rating sessions, not people: nothing
in the project says whether the same participant appears in several batches, so no annotator count is reported and no pseudo-ids were created.
Where: dataset description (batch structure; annotator counts not available). Source for the numbers: the raw file, checked with a one-off script.

## Ratings per text after the preprocessing (added 2026-10-29)

**24. Many texts end up with fewer than four ratings, and cannot be AU.**
A text needs four ratings to be AU (two groups of at least two). After the cleaning (which also drops ratings with an excluded group value), the
share of texts below four ratings is: popquorn 2 of 1,500 (0.1%), attitudes 41 of 625 (6.6%), MHS 919 of 5,990 (15.3%), Kumar 20,660 of 107,620
(19.2%). Ratings per text after the cleaning (min / median / max): popquorn 3 / 8 / 10, attitudes 2 / 6 / 11, MHS 2 / 4 / 777 (the 70 anchor comments, see
entry 2), Kumar 1 / 4 / 5. Those texts stay in the "all texts" columns of the EDA but are never in the AU set, which is one reason the AU share is
small. Source: `new_results/total_eda/ratings_per_text.csv` and `.txt`. Where: dataset descriptions.

**25. Share of toxic ratings differs a lot between datasets (after the cleaning).**
Ratings at or above the toxic threshold (3): popquorn 26.4%, attitudes 54.0%, MHS 42.4% (only "Yes" counts), Kumar 29.7%. Attitudes is also the only
dataset with a U-shaped distribution (31.1% rate 1 and 29.2% rate 5). Source: `new_results/total_eda/toxicity.csv`, `rating_distribution.csv`.

## Toxicity per group across datasets (added 2026-10-29)

**26. Group names were harmonized for the combined figures and table, which is a choice to state.**
`new_results/total_eda/toxicity_by_group.csv` and the two `toxicity_by_dimension_*.png` figures put the datasets side by side. Spellings are harmonized
(Man / male / Male -> man; Woman / female / Female -> woman; white / White -> White; black / Black / Black or African American -> Black) and the table
keeps the original names. The political dimensions are pooled as `politics` (attitudes `politics` left / center / right, Kumar `politics` liberal /
independent / conservative, MHS `ideology` liberal / neutral / conservative -> left / liberal, center / neutral / independent, right / conservative).
This treats three different measures as one: attitudes uses a numeric score cut at -0.25 and 0.25, Kumar a self-reported affiliation, MHS a seven-point
ideology collapsed to three. Education, occupation, religion and traditionalism exist in one dataset each, so they have no comparison.

**27. In the AU figure, many bars rest on few ratings.**
`AU ratings` per group (same csv): attitudes 11-41 for every group (e.g. gender: man 29, woman 24; politics center 11 and removed from the AU analysis),
popquorn 10-192, MHS 26-88, Kumar 272-3,434. Differences between groups in attitudes and popquorn are therefore noisy; do not read small gaps. Groups
removed from the AU analysis are left out of the AU figure (their all-texts numbers stay in the other figure and in the csv).
