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
