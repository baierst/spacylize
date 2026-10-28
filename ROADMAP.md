# Spacylize Roadmap

Spacylize distills the capabilities of large language models into compact, efficient spaCy models:
an LLM generates annotated training data, spacylize validates it, and a small standard spaCy
pipeline is trained on it.

**Guiding principles**

- **Local first** – local models (e.g. Ollama) are a first-class citizen, API-based providers are supported too.
- **Small footprint** – reuse what spaCy already provides wherever possible.
- **Simple CLI** – every step of the workflow is a single `spacylize` command.

**Legend:** ✅ done · 🚧 in progress / partial · 📋 planned

---

## ✅ Already implemented (v0.0.1)

### Foundation — Nov 2025
- ✅ Project setup with `pyproject.toml`, MIT license and package skeleton
- ✅ Typer-based CLI (`spacylize`) with `generate`, `visualize`, `validate`, `split`, `train` and `evaluate` commands
- ✅ First CLI tests

### LLM client — Dec 2025 (#1)
- ✅ `LLMClient`: unified access to OpenAI, Anthropic, Ollama and other providers via LiteLLM
- ✅ YAML-based LLM config (`model`, `api_key`, `api_base`, `max_tokens`) with pydantic validation
- ✅ `${ENV_VAR}` expansion in configs and `.env` support, so no keys live in config files

### Data generation & visualization — Dec 2025 (#2)
- ✅ `spacylize generate`: generates NER data from inline `[text](LABEL)` annotations and stores it as a spaCy `DocBin` (`.spacy`)
- ✅ `spacylize visualize`: serves generated samples in displaCy

### Validation — Dec 2025 (#3)
- ✅ `spacylize validate`: dataset report as JSON and PNG
  - number of documents, tokens and entities
  - tokens and entities per document (min / max / mean)
  - entity label distribution and entity length

### Docs & CI — Jan 2026 (#4 – #7)
- ✅ Sphinx API documentation, published to GitHub Pages
- ✅ GitHub Actions: Black formatting check and pytest on Python 3.10 – 3.12

### Text classification — Feb 2026 (#8)
- ✅ `generate --task textcat`, with a pluggable task registry (`TaskHandler`: parser + document builder per task)
- ✅ `validate`: textcat report with category distribution and document length; task auto-detection
- ✅ `visualize`: textcat view; task auto-detection
- ✅ Examples for both tasks: `examples/ner` and `examples/textcat`, each with an Ollama config (commented out)

### Prompt templates — Feb 2026 (#9)
- ✅ Structured prompt config (`task`, `entities` / `categories`, `domain`, `tone`, `length`, `language`, `constraints`, few-shot `examples`)
- ✅ Jinja2 templates per task render the system and user prompt
- ✅ Rendered prompts are saved next to the output (`system_prompt.txt`, `user_prompt.txt`) for review
- ✅ README configuration guide; tests for prompt config and templates

---

## Milestone 1 — End-to-end pipeline (v0.0.2)

*Goal: the README workflow works from generation to a trained, evaluated model, for NER and textcat.*

- 📋 **`split`** – implement `DataSpliter.run()`: shuffled train/dev split with seed and dev fraction
- 📋 **`train`** – implement `ModelTrainer.run()`: train NER and textcat pipelines from a base model,
  using spaCy's own config and training loop (`spacy init config` / `spacy train`)
- 📋 **`evaluate`** – implement `ModelEvaluater.run()`: precision, recall and F1 per label (NER),
  accuracy and macro F1 (textcat)
- 📋 Tests for splitter, trainer and evaluator
- 📋 Housekeeping
  - rename `ModelEvaluater` → `ModelEvaluator`, `DataSpliter` → `DataSplitter`
  - fix the README: Python 3.10+ (not 3.8+), example path `examples/ner/prompt.yaml` (not `examples/ecommerce/…`)
  - choose one package manager (uv or Poetry) and drop the other lockfile

## Milestone 2 — Local models & generation quality (v0.0.3)

*Goal: generating good, diverse data with a local model is the default experience.*

- 🚧 **Local models** – LiteLLM + `api_base` work with Ollama and the examples include a commented-out config; still to do:
  - 📋 test the Ollama setup end to end and make it the default in the examples
  - 📋 documentation for setting up local models, with recommended models
- 🚧 **Sample diversity** – the prompt is rendered once, so every sample gets the same prompt; still to do:
  - 📋 pass `temperature` from the prompt config to the LLM call (today it only changes a "Creativity" hint in the prompt text)
  - 📋 per-sample variation, e.g. value lists for `domain` / `tone` / `length` that are sampled for every request
- 📋 Report dropped annotations: count NER entities that don't align with token boundaries and textcat outputs that fail to parse
- 📋 Robustness: retries, rate limiting and a progress bar for large `--n-samples`

## Milestone 3 — Advanced validation (v0.0.4)

*Goal: check that generated data looks like real data, and find bad samples.*

- 📋 Validate several datasets at once (e.g. generated vs. real, labelled vs. unlabelled)
- 📋 Example datasets: a labelled and an unlabelled real-world reference set
- 📋 Similarity measures between datasets
  - vocabulary overlap
  - label distribution differences
  - embedding-based similarity (other measures to be evaluated)
- 📋 Topic check: flag samples that drift away from the configured `domain`
- 📋 Duplicate and near-duplicate detection

## Milestone 4 — Outlier visualization (v0.0.5)

*Goal: make problems in a dataset easy to see and fix.*

- 📋 Outlier detection on length, entity count and similarity, with the flagged samples saved to the report
- 📋 Outlier visualization: an interactive dashboard or PNG/JSON exports (to be decided)
- 📋 `visualize` options to browse flagged samples instead of only the first *n*

## Milestone 5 — Documentation & release (v0.1.0)

- 📋 Usage guides for each command, alongside the API reference
- 📋 Worked end-to-end examples for NER and textcat, with local and API models
- 📋 Publish to PyPI
