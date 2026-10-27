# perspective_audit

Tools to audit **whose perspective** a toxicity rating reflects, in two parts that feed each other:

1. **A-posteriori unimodality (AU).** Given a dataset in which many annotators rated the same texts, find the texts on which
   the annotators as a whole disagree (a *polarised* text) but **every annotator group is internally in agreement** (each group's
   ratings are unimodal). Those are the texts where the disagreement is explained by *who* the annotators are (gender, age,
   politics, ...), not by noise. The unimodality test is the normalized distance from unimodality (`ndfu`).
2. **Annotating those texts with language models.** Send the AU texts to several models (ChatGPT, DeepSeek, Mistral, Gemini,
   Llama, ...) with a prompt you write, and save one rating and one label per text and model, so the models' answers can be
   compared with the groups of human annotators.

Everything is a Python package (`src/perspective_audit`) plus one notebook (`notebooks/Datasets.ipynb`) that calls it.

- [Install](#install) · [Data](#data) · [Quick start](#quick-start) · [The AU rule](#the-au-rule)
- [Datasets](#datasets) · [Adding your own dataset](#adding-your-own-dataset) · [Saving the AU texts](#saving-the-au-texts)
- [Annotating with language models](#annotating-with-language-models) · [API keys](#api-keys) · [Choosing models](#choosing-models)
- [Writing the prompt](#writing-the-prompt) · [Running and reading results](#running-and-reading-results)
- [Other ways to prompt](#other-ways-to-prompt) · [Reproducibility](#reproducibility) · [Troubleshooting](#troubleshooting)
- [Package layout](#package-layout) · [Tests](#tests) · [Known limits](#known-limits)

---

## Install

Python 3.9 or newer.

```bash
cd perspective_audit
python -m venv .venv                 # optional but recommended
.venv\Scripts\activate               # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -e ".[all-models,test]"
```

`-e` (editable) means the notebook always runs the code in `src/`. The extras are:

| extra | installs | needed for |
|---|---|---|
| `openai` | `openai` | ChatGPT, Mistral, DeepSeek, and Llama through Ollama (all use the OpenAI client) |
| `gemini` | `google-genai` | Gemini |
| `oci` | `oci` | Grok on Oracle OCI (off by default) |
| `all-models` | the three above | everything |
| `test` | `pytest` | running the tests |

The base install (`numpy`, `pandas`, `scipy`, `ndfu`, `tqdm`) is enough for the AU analysis alone. If you call a model without its
extra installed you get an import error: install the extra and try again.

**Using the notebook.** Select a kernel whose Python has the package installed. If you installed it into a virtual environment,
register that environment as a kernel once:

```bash
pip install ipykernel
python -m ipykernel install --user --name perspective_audit
```

The first cell of the notebook checks that the package can be imported and tells you what to do if it cannot.

## Data

The raw datasets are **not in the repository** (some are hundreds of MB). Each dataset description in
`src/perspective_audit/datasets/<name>.py` says where its loader looks. `find_root()` finds the repository root by looking, from the
working directory upwards, for a folder named `datasets`; set the environment variable `THESIS_ROOT` to point elsewhere.

| dataset | file the loader reads | where it comes from |
|---|---|---|
| POPQUORN | `<root>/datasets/popquorn/popquorn.csv` | the POPQUORN release |
| Kumar et al. | `<root>/datasets/kumar/toxicity_ratings.json` (about 434 MB) | the Kumar et al. toxicity ratings release |
| Measuring Hate Speech | `<root>/datasets/mhs/train.parquet` if it exists, **otherwise downloaded** from Hugging Face (`ucberkeley-dlab/measuring-hate-speech`) | Hugging Face |
| Attitudes | `<root>/Thesis/Code/attitudes_embedded.csv`, or the path in `PERSPECTIVE_AUDIT_ATTITUDES_FILE` | the Attitudes data; **note:** despite the `.csv` name the file is a pickled `DataFrame`, and it is read with `pandas.read_pickle` |

Only read pickles from sources you trust. The stage that annotates texts with models does **not** need any raw dataset: it reads the
table written by the AU stage (`new_results/aposteriori_for_models.csv`).

## Quick start

### 1. Find the AU texts of a dataset

```python
from perspective_audit import SPECS, find_root, run_dataset

ROOT = find_root()
popquorn = run_dataset(SPECS["popquorn"], ROOT)

print(popquorn.describe())      # ratings and items before and after the cleaning
popquorn.groups                 # ratings per group after the cleaning
popquorn.au                     # the AU texts: one row per text and dimension
popquorn.per_dimension          # how many AU texts each dimension has
popquorn.per_group              # how many AU texts each group takes part in
popquorn.removal_cost           # what removing thin groups cost
```

`run_dataset` loads the dataset, cleans it (removed groups, prefilters), finds the AU texts of every dimension, removes thin groups,
and returns a `DatasetResult`.

### 2. Save them

```python
from perspective_audit import save_au_tables

save_au_tables({"popquorn": popquorn.au}, ROOT / "new_results")
```

### 3. Annotate them with a model

```python
from perspective_audit import ModelHub, annotate, load_model_inputs

inputs = load_model_inputs(ROOT / "new_results" / "aposteriori_for_models.csv")
hub = ModelHub.from_env()                      # reads your keys, see "API keys"

def my_prompt(texts, item):                    # you always write the prompt, see "Writing the prompt"
    return f"""Analyze the following {item}s for toxicity.
Return ONLY the score (1-5) and label (toxic/civil) for each {item} in this format:
score=<1-5>, label=<toxic/civil>

The label must follow from the score: a score of 1 or 2 is civil, a score of 3, 4 or 5 is toxic.

{item.capitalize()}s:
{texts}"""

annotate(hub["chatgpt"], inputs, ROOT / "new_results", prompt=my_prompt,
         run="baseline", datasets=["popquorn"])
```

The notebook does exactly this, one model and one dataset at a time, with switches at the top of section 6.

## The AU rule

Implemented in `perspective_audit.unimodality` (`Unimodality`, `PrunedAU`, `dfu_of_hist`).

A text is **AU for a dimension** when all of these hold:

1. It is **polarised**: the distance from unimodality (DFU) of all its ratings is greater than the tolerance (default 0).
2. At least **2 groups** of the dimension have **at least 2 ratings** each on that text.
3. **Every one** of those groups is itself unimodal (group DFU at most the tolerance).

A group with fewer than 2 ratings on a text is left out for that text and the others are kept. There is no minimum number of ratings per
text beyond that.

**Group label.** Each group gets `toxic` or `civil` from its ratings: the majority decides (a rating at or above `toxic_threshold` is
toxic); if the group is exactly half and half, its mean rating decides. Other rules are available: `label_rule="majority"` (ties are
labelled `tie`) and `label_rule="mean"`.

**Thin groups.** After the AU texts are found, a group that takes part in fewer than `min_au_texts` AU texts (default 10, per dataset
in the spec) is removed, smallest first, and the analysis is repeated. Its annotators stay in the data and still count in the overall DFU
of a text.

You can also use the tool directly on any long table of ratings:

```python
from perspective_audit import Unimodality

tool = Unimodality(scale=range(1, 6), min_ratings=2, min_groups=2, toxic_threshold=3, tolerance=0.0)
result = tool.au_pruned(table, "item_id", "rating", "gender", min_au_texts=10, keep_cols=["text"])
```

## Datasets

| name | texts | scale | toxic means | dimensions |
|---|---|---|---|---|
| `popquorn` | 1,500 | 1-5 | 3 or more | gender, race, age, education, occupation |
| `attitudes` | 626 tweets | 1-5 (`toyou`) | 3 or more | gender, race, age, politics, traditionalism |
| `mhs` | Measuring Hate Speech comments | 1-3 (`hatespeech`: No, Unclear, Yes) | Yes | gender, age, ideology |
| `kumar` | 107,620 comments, 5 ratings each | 1-5 (score 0-4 plus 1) | 3 or more | gender, race, age, politics, religion |

Age is grouped in all datasets as `gen z`, `millenial`, `gen x+` (the spelling `millenial` is the one used in the data and in the
thesis). The exact groups and what is removed are in each `datasets/<name>.py` and in the notebook sections 1-4. A rating with a value
that the spec removes (for example "Non-binary" in POPQUORN gender) is dropped before the analysis.

`SPECS` is a dict `{name: DatasetSpec}`; `get_spec(name)` looks one up.

## Adding your own dataset

A dataset is described by one `DatasetSpec`. You write a loader that returns a **long table with one row per rating**, and say how each
annotator attribute maps to groups.

```python
import pandas as pd
from perspective_audit import DatasetSpec, run_dataset

def load(root):
    raw = pd.read_csv(root / "datasets" / "mine" / "ratings.csv")
    return pd.DataFrame({
        "item_id": raw["comment_id"].astype(str),     # required
        "text":    raw["comment"],                    # required
        "rating":  raw["toxicity"],                   # required, 1..scale_max
        "gender":  raw["rater_gender"],               # the raw attribute columns
        "age":     raw["rater_age"],
    })

spec = DatasetSpec(
    name="mine",
    load=load,
    dimensions={                                       # dimension -> (raw column, mapping)
        "gender": ("gender", {"Man": "man", "Woman": "woman"}),     # a value not listed is removed
        "age":    ("age", lambda s: pd.cut(pd.to_numeric(s, errors="coerce"), [0, 24, 44, 120],
                                           labels=["gen z", "millenial", "gen x+"]).astype(object)),
    },
    scale_max=5,
    toxic_threshold=3,
    min_au_texts=10,
    item_word="comment",                               # how a text is called in prompts
)

result = run_dataset(spec, ROOT)
```

A mapping is either a dict (a raw value not in the dict is removed) or a function from a Series to a Series (a missing result is
removed). Optional fields: `tolerance` (DFU tolerated as "unimodal"), `prefilter` (a function applied to the raw table first, for example
"keep texts with at least 4 ratings"), and `description`. To make it available by name, add it to the `SPECS` dict (`SPECS["mine"] = spec`), or, to make it permanent, add a module
like the ones in `datasets/` and list it in `datasets/__init__.py`.

## Saving the AU texts

`save_au_tables` writes, under the folder you give it (the notebook uses `new_results/`):

| file | content |
|---|---|
| `<name>/<name>_au.csv` | the AU texts of one dataset |
| `aposteriori_all.csv` | all datasets together |
| `<name>/<name>_au_sample.csv` | a sample of a large dataset, if you ask for one |
| `aposteriori_for_models.csv` | the texts to give to the models |

One row is a **text in one dimension**: `row_id`, `dataset`, `dimension`, `item_id`, `text`, `n_ratings`, `dfu`, `toxic_pole` (the label of
each compared group), `kept_groups`, `group_sizes` and `group_label` (the last three are JSON). A text that is AU in several dimensions has
one row per dimension. Read these files back with `read_au_csv`, which turns the JSON columns into lists and dicts again.

**Samples for large datasets.** Kumar has far more AU texts than the other datasets together, so it is used as a reference and only a
sample goes to the models. With `method="cover"` the sample is the **smallest set of AU texts in which every group of every dimension
appears at least `k` times** (an exact integer program, fixed `seed`, so the same input gives the same file). With `method="random"` it is `n`
random texts per dimension. The cover sample is deliberately not proportional: use it to make sure every group is seen, not to estimate
how common something is.

```python
save_au_tables(
    {name: r.au for name, r in results.items()}, ROOT / "new_results",
    exclude_for_models=("kumar",),                   # kumar contributes its sample, not its full table
    sample={"kumar": {"method": "cover", "k": 10, "seed": 42, "dimensions": list(SPECS["kumar"].dimensions)}},
)
```

## Annotating with language models

`annotate` takes the texts of every dataset, sends them to **one model**, in batches (10 texts per call by default, a batch never mixes
datasets), and saves what comes back. Everything is organised **per dataset** and **per run**:

```
new_results/
  run_log.csv                          one line per pass: model, exact model id, settings, time, status
  <dataset>/
    <run>/
      <model>.csv                      one row per text, see below
      all_models.csv                   the models side by side
      raw/<model>.raw.jsonl            every prompt and reply, batch by batch (the source of truth)
```

**`<model>.csv`** has one row per text:

| column | meaning |
|---|---|
| `dataset`, `item_id`, `text` | which text (`item_id` links back to the AU table, which holds the dimension and groups) |
| `model`, `run` | who rated it and in which run |
| `score` | the model's rating, 1 to 5 |
| `label` | `civil` for a score of 1-2, `toxic` for 3-5 |

The **label always follows from the score**, whatever label the model wrote, so the two never disagree. A reply without a valid score
counts as unusable.

**`all_models.csv`** has one row per text and `<model>_score`, `<model>_label` for every model run so far; it is refreshed after each
model.

## API keys

Keys are read from environment variables, or from a `.env` file. The lookup order is:

1. real environment variables;
2. the first `.env` file found in the working directory or any parent folder.

To set up:

```bash
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

then fill in the values. **Never commit `.env`** (it is in `.gitignore`).

| model | variable | notes |
|---|---|---|
| ChatGPT | `chat_api` | an OpenAI API key |
| Gemini | `gemini_key` | a Google AI Studio key |
| Mistral | `mistral_key` | a Mistral API key |
| DeepSeek | `deepseek_key` | a DeepSeek API key |
| Llama | none | runs **locally** through [Ollama](https://ollama.com): install it, then `ollama pull llama3:8b` and leave it running |
| Grok (off by default) | `OCI_KEY`, `OCI_CONFIG`, `COMPARTMENT_ID`, `MODEL_ID` | Oracle OCI; a value starting with `@` is read from that file |

The variable names are the ones in `.env.example`. If you prefer other names (for example `OPENAI_API_KEY`), set
`CHATGPT_API_KEY_ENV=OPENAI_API_KEY` in `.env`: the model then looks the key up under that name.

**Check what is set** without calling any model:

```python
hub = ModelHub.from_env()
hub.available()     # models that can be called
hub.missing()       # {'gemini': 'gemini_key is not set', ...}
```

A model whose key is missing stays in the list but raises `gemini_key is not set (looked in the environment and in <path>)` when called.

## Choosing models

The keys do **not** choose the models; they only decide which of the chosen models can run. `ModelHub.from_env()` builds the list in this
order:

1. names passed in code: `ModelHub.from_env(["chatgpt", "gemini"])`;
2. otherwise `PERSPECTIVE_AUDIT_MODELS=chatgpt,gemini` in `.env` or the environment;
3. otherwise every model in `DEFAULT_CONFIGS` that is `enabled`: `chatgpt`, `llama`, `gemini`, `mistral`, `deepseek` (not `grok`).

An unknown name raises an error that lists the known ones. Then use `hub["chatgpt"]` for one model, or loop over `hub`.

**Defaults** (in `DEFAULT_CONFIGS`, `models.py`):

| name | model id | endpoint | temperature | max tokens |
|---|---|---|---|---|
| `chatgpt` | `gpt-4o-mini` | OpenAI | 0 | 256 |
| `deepseek` | `deepseek-chat` | `https://api.deepseek.com` | 0 | 256 |
| `mistral` | `open-mistral-7b` | `https://api.mistral.ai/v1` | 0 | 256 |
| `gemini` | `gemini-2.5-flash` (thinking off) | Google GenAI | 0 | 512 |
| `llama` | `llama3:8b` | `http://localhost:11434/v1` (Ollama) | 0, retried higher after a refusal | 256 |

**Change a setting without touching the code**, with variables in `.env` (name in capitals):

```
GEMINI_MODEL=gemini-2.5-pro
LLAMA_BASE_URL=http://localhost:11434/v1
MISTRAL_TEMPERATURE=0
DEEPSEEK_MAX_TOKENS=512
CHATGPT_API_KEY_ENV=OPENAI_API_KEY
```

The supported suffixes are `_MODEL`, `_BASE_URL`, `_API_KEY_ENV`, `_TEMPERATURE`, `_MAX_TOKENS`.

**Add a model.** Anything with an OpenAI-style chat API (Together, Groq, vLLM, an Ollama model, ...) needs only a config:

```python
from perspective_audit import ModelConfig, ModelHub

configs = {"qwen": ModelConfig("qwen", "openai_compatible", "qwen2.5:7b-instruct",
                               api_key_env=None, base_url="http://localhost:11434/v1")}
hub = ModelHub.from_env(["qwen"], configs=configs)
```

`ModelConfig` also has `retries` (attempts when a call raises; rate-limit errors wait longer), `retry_temperatures` and `reply_pattern` (retry
at the next temperature until the reply matches the pattern, used for Llama refusals) and `request_extra` (provider specific options).
A provider with its own API, not OpenAI-compatible, needs a small class in `models.py` next to `Gemini` and `OCI`.

## Writing the prompt

**You always pass the prompt**, as a function. The package has no hidden prompt, so what was sent is exactly what you wrote.

```python
def my_prompt(texts, item):
    return f"""Analyze the following {item}s for toxicity.
Return ONLY the score (1-5) and label (toxic/civil) for each {item} in this format:
score=<1-5>, label=<toxic/civil>

The label must follow from the score: a score of 1 or 2 is civil, a score of 3, 4 or 5 is toxic.

{item.capitalize()}s:
{texts}"""
```

- `texts` is the batch **already numbered** (`1. ...`, `2. ...`), so the answers can be matched to the texts.
- `item` is the word for a text in that dataset (`"tweet"`, `"comment"`, ...), set by `item_word` in the dataset spec. Override it with
  `annotate(..., item_words={"kumar": "post"})`.
- Return the whole text that is sent to the model.

**The reply format is fixed.** The parser reads one line per text of the form `score=<n>` (and ignores any label the model wrote). Your
prompt must ask for that, one line per text, in the order of the texts. A prompt that asks for another format will not parse.

**The run name is the condition, not the prompt.** `run="baseline"` means no persona; persona runs get their own names. Changing the
wording of the prompt does not change the run name. Mind that saved batches are resumed per run name: if you change the prompt and
re-run `baseline` as it is, the batches already saved are skipped and only the missing ones are asked with the new prompt, so
answers of two prompts would end up in one run. To re-run everything with a new prompt, delete that run's folder
(`<dataset>/baseline/`) first. The raw file records the exact prompt of every batch, so you can always see which prompt produced which answer.

The function's name is recorded as `prompt_version` in `run_log.csv`, and the exact text sent for every batch is saved in the raw file.

## Running and reading results

```python
annotate(model, inputs, out_dir, prompt, run="baseline", batch_size=10, parse_retries=2,
         limit=None, datasets=None, item_words=None, scale_max=5, toxic_threshold=3, show_progress=True)
```

| argument | meaning |
|---|---|
| `model` | one `Model` from the hub |
| `inputs` | `{dataset: table with item_id, text}`, from `load_model_inputs(path)` (each text once) |
| `out_dir` | where the dataset folders go (the notebook uses `new_results/`) |
| `prompt` | your function `prompt(texts, item)` |
| `run` | the condition, e.g. `baseline` (no persona); results go to `<dataset>/<run>/` |
| `limit` | only the first N texts of each dataset: a **pilot** |
| `datasets` | which datasets to run (default: all in `inputs`) |
| `batch_size` | texts per model call |
| `parse_retries` | how many times a batch with the wrong number of scores is asked again before the run stops |
| `scale_max`, `toxic_threshold` | the scale and cutoff (5 and 3 by default) |
| `show_progress` | the progress bar |

**Progress bar.** A `tqdm` bar per dataset and model shows texts done, elapsed time, time left, and the batch being asked with the start of its
first text. It renders as a widget in a notebook and as a text bar in a terminal.

**Stopping and resuming.** A run that stops (an error, an interrupt, a lost connection) keeps every batch it finished. Run the same call
again, with the same `run`, and it **continues at the first missing batch**. If the saved batches do not match the current input (the texts
changed), it refuses; use another run name or delete the run's folder.

**A batch that cannot be parsed** (the reply does not have exactly one score per text) is asked again `parse_retries` times; if it still fails,
the run stops with an `AnnotationError` and the reply is kept in the raw file so you can see what the model said.

**Looking at the results**

```python
from perspective_audit import ensemble, inspect_run, progress, collect_runs

inspect_run(out_dir, "chatgpt", "baseline")       # per dataset: texts, mean score, share called toxic, batches asked again / not parsed
ensemble(out_dir, "popquorn", "baseline")         # the models side by side (also written to all_models.csv)
progress(inputs, out_dir, hub.names, run="baseline")   # texts with a saved score, per dataset and model
collect_runs(out_dir, "baseline")                 # every saved row of a run, in one table
```

**Prompt sensitivity on a subset.** To see whether the answers depend on the wording, annotate the same random subset with another prompt,
in its own run, and compare it with the baseline answers for the same texts:

```python
from perspective_audit import sample_inputs

subset = sample_inputs(inputs, 50, seed=42)        # 50 random texts per dataset; the same n and seed give the same texts
annotate(model, subset, out_dir, prompt=other_prompt, run="prompt_sensitivity")      # -> <dataset>/prompt_sensitivity/, next to <dataset>/baseline/
```

The baseline is not asked again: its file already has these texts. This is section 6.2 of the notebook, where each dataset ends up with a
`baseline/`, a `prompt_sensitivity/` and a `prompt_sensitivity_2/` folder (two rewordings), and figures and tables compare them. The exact prompts are written to `new_results/prompts.md`.

**Agreement between two sets of ratings.** Three measures, for any two sets of ratings of the same texts (two prompts of one model, two
models, a model and a group of people). 1 means perfect agreement, 0 means no more than chance, and a missing value is left out:

```python
from perspective_audit import cohen_kappa, weighted_kappa, spearman

weighted_kappa(scores_a, scores_b)        # scores 1-5: a big disagreement counts more than a small one (quadratic weights)
cohen_kappa(labels_a, labels_b)           # labels such as "toxic" / "civil", corrected for the agreement chance alone would give
spearman(scores_a, scores_b)              # do the two sets put the texts in the same order?
```

Prefer the kappas to the plain share of equal labels: when almost everything is toxic (as in MHS), most pairs agree by chance.

What to check after a model finishes: `texts` is what you expect, `mean score` and `toxic %` are plausible (all 1s or all 5s means the model
is not following the prompt), and `batches asked again` and `batches not parsed` are 0 or small.

**A good order:** run a pilot first (`limit=10`, `run="baseline_pilot"`: one call, almost free), look at the table, then the full run
(`limit=None`, `run="baseline"`).

**Cost and time.** Cloud models take about a second per batch of 10. A local model is much slower if it does not fit in GPU memory: for
example `llama3:8b` on a 4 GB GPU runs partly on the CPU and takes several seconds per batch, and each refusal retried at a higher
temperature costs another full call.

## Other ways to prompt

These are in `perspective_audit.scoring` and are independent of `annotate`.

**`score_texts(model, texts, prompt=None, batch_size=10, parse_retries=2)`** scores a list of texts and returns a table (`batch`,
`text_index`, `text`, `score`, `label`). It uses a `ScoringPrompt`, which has the thesis wording (`"v1"`) and a variant that also states the
threshold (`"v2"`).

**`score_personas(model, items, persona_map, prompt=None)`** asks the model to answer *as* each group of one dimension (persona
prompting). `items` has the columns `text` and `dimension`; `persona_map` gives the personas of each dimension, for example
`{"gender": ["man", "woman"]}`. The reply is JSON keyed by persona. It returns one row per item and persona (`row`, `dimension`, `persona`,
`label`, `score`). A reply is accepted only when every persona has a valid score; otherwise the call is repeated with a larger token budget
(truncation is the usual failure), and items still incomplete after `max_attempts` raise an error that lists them. `PersonaPrompt` has a long
wording (`"v1"`) and a short one (`"compact"`).

## Reproducibility

- **Deterministic AU analysis.** The AU rule has no randomness. The only random step is the sample of a large dataset, and it takes a
  fixed `seed` (42 in the notebook).
- **Everything sent and received is saved.** `raw/<model>.raw.jsonl` has the exact prompt, the reply, the parsed scores, the number of
  attempts and a timestamp for every batch.
- **Every pass is logged.** `run_log.csv` records the time, run, dataset, model, provider, **exact model id**, temperature, token limit,
  prompt name, item word, batch size, number of texts, batches, batches resumed, model calls, seconds and status. Cloud models can change
  behind the same name, so the log is the record of what was actually used.
- **The CSV is rebuilt from the saved batches**, never from the current input, so it always matches what was asked.
- **Temperature 0 by default**, but a hosted model can still give different answers on different days; the raw files are the record of
  what you got.
- **Do not mix prompts in one run.** The run name is the condition, so when you change the prompt, delete the run's folder and run it
  again. Every batch in the raw file keeps its exact prompt.
- Pin your environment (`pip freeze > requirements.txt`) if you need to reproduce numbers exactly.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: perspective_audit` in the notebook | the notebook kernel's Python does not have the package. Select the right kernel, or `pip install -e .` in that Python |
| `TypeError: annotate() got an unexpected keyword argument ...` or a function that "does not exist" after you changed the code | the kernel loaded the package before the change. **Restart the kernel** and run the setup cell again |
| `NameError: name 'all_models' is not defined` | the setup cell of section 6 has not been run in this kernel. Run it first |
| `chat_api is not set (looked in ...)` | the key is missing. Add it to `.env`, or set the variable; `hub.missing()` lists what is missing |
| `ModuleNotFoundError: openai` (or `google`) | the extra is not installed: `pip install -e ".[all-models]"` |
| `FileNotFoundError: Could not find a folder containing 'datasets'` | run from inside the repository, or set `THESIS_ROOT` |
| `AnnotationError: ... expected 10 scores, received 9` | the model's reply did not have one `score=<n>` line per text. Open the reply in `raw/<model>.raw.jsonl`; usually the prompt needs to be stricter, or the model adds text around the scores |
| `AnnotationError: saved batch N holds other texts than the input` | the input changed since this run was started. Use a new `run` name |
| Llama is very slow, or a request ends with HTTP 500 | the model does not fit in GPU memory, or a refusal is being retried, or you stopped the cell (Ollama logs a stopped client as a 500). Watch the progress bar: about 10 texts every few seconds is normal |
| Ollama connection refused | start Ollama and `ollama pull llama3:8b` |
| a model's results look odd (all 5s) | read its raw replies; it may not be following the prompt. The scores are saved as the model gave them |

## Package layout

```
perspective_audit/
  pyproject.toml
  .env.example                  the keys and optional settings, to copy to .env
  notebooks/Datasets.ipynb      the thesis notebook: AU texts of four datasets, saving, and annotating with models
  src/perspective_audit/
    unimodality.py              DFU and the AU rule (Unimodality, PrunedAU)
    spec.py                     DatasetSpec
    datasets/                   popquorn.py, attitudes.py, mhs.py, kumar.py, groups.py (shared age groups)
    pipeline.py                 clean, run_dataset, DatasetResult
    export.py                   save_au_tables and the CSV format of an AU table
    sampling.py                 minimal_cover, cover_per_dimension (the sample of a large dataset)
    env.py                      the .env file and environment variables
    models.py                   ModelConfig, Model, ModelHub, DEFAULT_CONFIGS and the providers
    prompts.py                  ScoringPrompt, PersonaPrompt and the parsers of their answers
    scoring.py                  score_texts, score_personas
    annotation.py               annotate, ensemble, inspect_run, progress, collect_runs, load_model_inputs
    utils.py                    find_root, read_au_csv and small helpers
  tests/                        tests with stand-in models (no keys, network or credits)
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests use stand-in models, so they need no keys, no network and cost nothing. They cover the AU rule, the sampling, the export format,
the models (retries, key handling, parsing), and `annotate` (the files it writes, resuming, unparseable replies, custom prompts, the run log).

## Known limits

- **The reply format of `annotate` is fixed** (`score=<n>` per line). The prompt is yours, the parser is not.
- **The labels are `toxic` and `civil`.** The rest of the package (summaries, the ensemble table) depends on those two strings.
- **`annotate` runs one model on one dataset at a time**, on purpose, so a problem shows where it happens. Loop over `hub` to run all.
- **Key variable names** (`chat_api`, `gemini_key`, ...) are the ones of the thesis. Rename them with `<NAME>_API_KEY_ENV`.
- **Persona prompting** (`score_personas`) takes a `PersonaPrompt` object, not a function like `annotate`.
- **The Attitudes loader reads a pickle** from a fixed location (or `PERSPECTIVE_AUDIT_ATTITUDES_FILE`); the other loaders need the files
  of the [Data](#data) section in place.
