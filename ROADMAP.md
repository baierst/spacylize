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

## Phase 1 — Data Creation & Validation

- ✅ LLM client for API-based providers (OpenAI, Anthropic, …) with YAML config
- 🚧 Local models (Ollama) as the default: test the setup end to end and document it
- ✅ Structured prompt templates; rendered prompts saved for review
- ✅ Training data generation for NER and text classification
- 📋 Sample diversity: vary the prompt for each sample
- ✅ Validation report (JSON and PNG): document length and label distribution
- 📋 Interactive validation dashboard: embedding map, outliers, duplicates, diversity and label quality
- 📋 Topic check against the configured domain
- ✅ Browse generated samples (`visualize`)

## Phase 2 — Training, Evaluation & Iteration

- 📋 Train/dev split
- 📋 Training interface, with saved hyperparameters so runs can be reproduced
- 📋 Evaluation with per-label metrics
- 📋 Feedback loop: evaluation points to weak labels and guides targeted data generation
- 📋 Real-world reference datasets (labelled and unlabelled) for evaluation and comparison
- 📋 Documentation for the CLI and the Python library, with a realistic end-to-end example
- 📋 Release on PyPI
