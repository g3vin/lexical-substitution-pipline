# Context-Aware Synonym Generation Pipeline via Multi-Stage Neural Filtering and Reranking

This project is a generates context-aware Synonyms. Given a sentence and a target word, it generates and ranks single-word substitutes that fit into its context.

Take a common example:

```text
Sentence: The dog chased the cat.
Target: chased
Output: pursued, followed, hunted, ...
```

This pipeline combines:

- lexical resources such as `WordNet` and optionally `SWORDS`
- neural candidate generation with FLAN-T5 and masked language modeling
- linguistic filtering with `spaCy`
- semantic similarity scoring with `Sentence-BERT`
- optional substitute-quality scoring
- optional contradiction filtering
- optional cross-encoder reranking

The core code of the project lives in `pipeline/`, but it also includes evaluation, training, and reporting scripts built around the SWORDS dataset

## Install

For your convience, and my own sanity, this repo includes:

- a packaged installer in `pyproject.toml`
- a pinned lockfile in `requirements.lock`
- a `Makefile` for common workflows

### Recommended setup

```bash
python -m venv .venv
source .venv/bin/activate
make install-editable
make resources
```

### Installed commands

- `lexsub`: main inference CLI
- `lexsub-eval`: SWORDS evaluation and tuning
- `lexsub-plot-report`: report plots
- `lexsub-train-substitute`: substitute-quality model training

Repo-local entry points like `python pipeline.py` and `python evaluate_swords.py` also still work

## Quick Start

Try the Demo:

```bash
lexsub
```

Run on your own sentence:

```bash
lexsub "The dog chased the cat." "chased"
```

If the target appears multiple times, pass the character offset feild:

```bash
lexsub "He left after he left the room." "left" --target-offset 3
```

Maybe you want to print only substitute words:

```bash
lexsub "The dog chased the cat." "chased" --compact
```

Slow device? Try a faster configuration:

```bash
lexsub "The dog chased the cat." "chased" --generation-mode fast --no-reranker
```

## Make Commands

```bash
make help
make install-editable
make resources
make demo
make demo-fast
make batch
make eval
make tune
make train-substitute
make train-ranker
make plots
make plots-model
```

Useful overrides:

```bash
make eval TOP_K=20 SWORDS_SPLIT=test
make plots-model TOP_K=10 REPORT_DIR=report_plots_model
make tune RANKER_CONFIG=ranker_config.json
```

## Main CLI

The main inference command is:

```bash
lexsub "sentence" "target" [options]
```

Common options you should know:

- `--top-k`: number of results to return
- `--generation-mode`: one of `full`, `balanced`, `fast`, `resources`, `mlm`, `t5`
- `--no-reranker`: disable the cross-encoder reranker
- `--substitute-model`: path to the trained substitute model
- `--substitute-hard-filter`: use substitute quality as a hard filter
- `--ranker-config`: custom ranker config JSON
- `--learned-ranker-model`: learned ranker artifact trained from SWORDS features
- `--no-ranker-config`: use code defaults instead of the bundled/tuned config
- `--no-swords`: disable SWORDS lexical candidates
- `--no-morphology`: disable surface-form matching
- `--compact`: print only ranked words

Recommended modes:

- `balanced`: best default tradeoff in this repo
- `fast`: quicker, lower-cost runs
- `resources`: lexical baseline
- `full`: most expensive search

## Evaluation, Training, and Reporting

### Evaluation

Run evaluation on SWORDS:

```bash
lexsub-eval --split dev --top-k 10
```

Evaluate with the included substitute model:

```bash
lexsub-eval --split dev --substitute-model ./substitute-roberta
```

Tune and save a ranker config:

```bash
lexsub-eval \
  --split dev \
  --substitute-model ./substitute-roberta \
  --tune \
  --output-config ranker_config.json
```

The evaluator reports ranking metrics such as `NDCG@k`, `MAP@k`, `Precision@k`, and pairwise accuracy

### Train the substitute-quality model

Wanna re-train with some new parameters?

```bash
lexsub-train-substitute --output-dir ./substitute-roberta
```

What about with a new base model?

```bash
lexsub-train-substitute \
  --model-name roberta-base \
  --split dev \
  --output-dir ./substitute-roberta
```

### Train the learned ranker

If you want to train a grouped learned ranker directly from SWORDS:

```bash
lexsub-train-ranker \
  --ranker-config ./ranker_config.json \
  --substitute-model ./substitute-roberta \
  --output-model ./learned_ranker.pkl
```

Or train from an existing feature table:

```bash
lexsub-train-ranker \
  --features-csv ./report_plots_model/model_candidate_features.csv \
  --ranker-config ./ranker_config.json \
  --output-model ./learned_ranker.pkl
```

### Reporting and plots

I've included some built in plotting for evaluation and debugging purposes.

Dataset-only plots:

```bash
lexsub-plot-report --split dev --output-dir report_plots
```

Full report plots with model behavior:

```bash
lexsub-plot-report \
  --split dev \
  --include-model-plots \
  --substitute-model ./substitute-roberta \
  --ranker-config ./ranker_config.json \
  --learned-ranker-model ./learned_ranker.pkl
```

## Python API

You can also use the pipeline directly, instead of through the CLI:

```python
from pipeline import LexicalSubstitutionPipeline

pipeline = LexicalSubstitutionPipeline(
    generation_mode="balanced",
    substitute_classifier_model_name="./substitute-roberta",
    substitute_positive_label="valid",
    ranker_config_path="./ranker_config.json",
)

results = pipeline.substitute(
    "The dog chased the cat.",
    "chased",
    top_k=5,
)

for candidate in results:
    print(candidate.word, candidate.final_score)
```

You can also score a candidate list you have made (or perhaps that another model has generated):

```python
scored = pipeline.score_substitutes(
    "The dog chased the cat.",
    "chased",
    ["followed", "pursued", "ran", "ignored"],
)
```

## Included Artifacts

- `substitute-roberta/`: trained substitute-quality model
- `ranker_config.json`: tuned ranker configuration
- `data/swords/`: local SWORDS dev/test files
- `example_outputs.json`: example batch output
- `report_plots_model/`: generated report figures

You should be able to run the pipeline without retraining anything first!!

## Constraints and Future Work

There's a few practical limits of this project you should keep in mind:

- the system focuses on single-word substitutes, not multi-word paraphrases (maybe I'll add this in the future lol)
- this should be common sense, but the target must appear in the sentence
- repeated targets require `--target-offset` for best results