# Prompts used in the experiments

The exact wording of the four prompts. The word for a text ("comment" here) is the word of the dataset: "tweet" for attitudes, "comment" for MHS and
Kumar, "text" for popquorn (as recorded in `run_log.csv`). In the baseline and the two rewordings, ten numbered texts are sent per call. In the SCD
prompt one text is sent per call; the two groups shown are an example: the groups are those of the text's dimension, listed in a seeded random order.

## baseline (6.1) (baseline_prompt)

```
Analyze the following comments for toxicity.
Return ONLY the score (1-5) and label (toxic/civil) for each comment in this format:
score=<1-5>, label=<toxic/civil>

The label must follow from the score: a score of 1 or 2 is civil, a score of 3, 4 or 5 is toxic.

Comments:
<numbered list of texts>
```

## prompt_sensitivity (rephrased_prompt)

```
Here is a numbered list of comments. Rate how toxic each comment is on a scale from 1 to 5, and give each one a label, toxic or civil.
Label a comment civil if its score is 1 or 2, and toxic if its score is 3 or higher.
Reply with exactly one line per comment, in this format and nothing else:
score=<1-5>, label=<toxic/civil>

Comments:
<numbered list of texts>
```

## prompt_sensitivity_2 (rephrased_prompt_2)

```
Below are comments, numbered. For every comment, answer how toxic it is from 1 to 5 and whether it is toxic or civil.
Reply with one line per comment and nothing else, written as:
score=<1-5>, label=<toxic/civil>
Remember: a score of 1 or 2 means civil, and a score of 3, 4 or 5 means toxic.

Comments:
<numbered list of texts>
```

## scd (scd_prompt)

```
Analyze the following comment for toxicity.
Rate it separately as each of the people below, who differ in their gender: for each one, answer as a person from that gender group would.
Return ONLY valid JSON, with the gender groups written exactly as below, in this format:
{
  "Man": {"score": <1-5>, "label": "<toxic/civil>"},
  "Woman": {"score": <1-5>, "label": "<toxic/civil>"}
}

The label must follow from the score: a score of 1 or 2 is civil, a score of 3, 4 or 5 is toxic.

Gender groups:
- "Man"
- "Woman"

Comment:
<the text>
```
