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
- ✅ YAML prompt config with system and user message
- ✅ `spacylize generate`: generates NER data from inline `[text](LABEL)` annotations and stores it as a spaCy `DocBin` (`.spacy`)
- ✅ `spacylize visualize`: serves generated NER samples in displaCy
- ✅ E-commerce example (`examples/ecommerce`): product attribute extraction with 7 entity labels

### Validation — Dec 2025 (#3)
- ✅ `spacylize validate`: dataset report as JSON and PNG
  - number of documents, tokens and entities
  - tokens and entities per document (min / max / mean)
  - entity label distribution and entity length

### Docs & CI — Jan 2026 (#4 – #7)
- ✅ Sphinx API documentation, published to GitHub Pages
- ✅ GitHub Actions: Black formatting check and pytest on Python 3.10 – 3.12
- ✅ README with an end-to-end usage example

---

## Milestone 1 — End-to-end NER pipeline (v0.0.2)

*Goal: the README workflow works from generation to a trained, evaluated model.*

- 📋 **`split`** – implement `DataSpliter.run()`: shuffled train/dev split with seed and dev fraction
- 📋 **`train`** – implement `ModelTrainer.run()`: train a spaCy NER pipeline from a base model, using
  spaCy's own config and training loop (`spacy init config` / `spacy train`)
- 📋 **`evaluate`** – implement `ModelEvaluater.run()`: precision, recall and F1, overall and per label
- 📋 Report entities dropped during generation (when an annotation does not align with token boundaries)
- 📋 Tests for splitter, trainer, evaluator and the config modules
- 📋 Housekeeping
  - rename `promt.yaml` → `prompt.yaml`, `ModelEvaluater` → `ModelEvaluator`, `DataSpliter` → `DataSplitter`
  - fix the README: Python 3.10+ (not 3.8+), `.spacy` output path in the generate step
  - choose one package manager (uv or Poetry) and drop the other lockfile

## Milestone 2 — Local models & better prompts (v0.0.3)

*Goal: generating good, diverse data with a local model is the default experience.*

- 🚧 **Local models** – LiteLLM + `api_base` already make Ollama possible; still to do:
  - 📋 example `llm.yaml` for Ollama, tested end to end
  - 📋 documentation for setting up local models
  - 📋 recommended local models for data generation
- 🚧 **Prompt templates** – today every sample uses the same fixed prompt; still to do:
  - 📋 template variables (e.g. `{labels}`, `{topic}`, `{style}`) filled per sample
  - 📋 lists of values for variables to sample from, for more diverse data
  - 📋 few-shot examples in the prompt config
- 📋 Robustness: retries, rate limiting and progress output for large `--n-samples`

## Milestone 3 — Text classification (v0.0.4)

*Goal: a second task type next to NER.*

- 📋 `generate --task textcat`: prompt format and parser for document labels (`doc.cats`)
  (today `--task` is accepted but ignored)
- 📋 `validate`: textcat report with label distribution and document length per label
- 📋 `visualize`: textcat support
- 📋 `train` / `evaluate`: textcat pipelines and metrics
- 📋 Example: a text classification dataset next to `examples/ecommerce`

## Milestone 4 — Advanced validation (v0.0.5)

*Goal: check that generated data looks like real data, and find bad samples.*

- 📋 Validate several datasets at once (e.g. generated vs. real, labelled vs. unlabelled)
- 📋 Example datasets: a labelled and an unlabelled real-world reference set
- 📋 Similarity measures between datasets
  - vocabulary overlap
  - label distribution differences
  - embedding-based similarity (other measures to be evaluated)
- 📋 Topic check: flag samples that drift away from the intended topic
- 📋 Outlier detection on length, entity count and similarity, with the flagged samples saved to the report
- 📋 Outlier visualization: an interactive dashboard or PNG/JSON exports (to be decided)

## Milestone 5 — Documentation & release (v0.1.0)

- 📋 Usage guides for each command, alongside the API reference
- 📋 Worked examples for NER and textcat, with local and API models
- 📋 Publish to PyPI
