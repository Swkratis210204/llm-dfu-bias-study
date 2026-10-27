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
