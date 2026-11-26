# Spacylize
**Documentation:** https://baierst.github.io/spacylize

Spacylize distills large language models into compact spaCy models: an LLM generates annotated training
data, spacylize validates and cleans it, and a small spaCy pipeline is trained on it.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run spacylize --help
```

The examples use the local model `gemma4` via [Ollama](https://ollama.com): start Ollama (the app, or
`ollama serve`) and run `ollama pull gemma4`. To use a cloud model instead, switch to one of the
commented-out models in `llm.yaml` and set its API key (e.g. `ANTHROPIC_API_KEY`) in your environment or
a `.env` file.

```yaml
# llm.yaml
model: ollama_chat/gemma4         # any Ollama model: ollama_chat/<name>
api_base: http://localhost:11434
max_tokens: 1024                  # optional, default: 1024
think: false                      # optional, default: model default; gemma4 is much faster without thinking
```

For development: `uv sync --group dev`, then `uv run black --check .` and `uv run pytest`.

## Prompt configuration

A `prompt.yaml` describes what to generate; templates turn it into the prompts. The prompt is rendered for
every sample, so samples vary. NER example:

```yaml
task: ner
entities: [PRODUCT_TYPE, BRAND, MODEL, COLOR]
min_entities: 1                  # optional, default: 1; each sample asks for a random subset of entities
domain: "tech product descriptions"
tone:                            # domain, tone and length: a text, or a list (one picked per sample)
  - "professional marketing copy"
  - "concise and factual"
length: "2-3 sentences"
language: "en"                   # optional, default: "en"
temperature: 0.7                 # optional, default: provider default
constraints:                     # optional
  - "Use realistic, coherent values"
examples:                        # optional, few-shot examples
  - text: "The new [iPhone 15](MODEL) by [Apple](BRAND) is a [Smartphone](PRODUCT_TYPE) in [Black](COLOR)."
```

For text classification, use `task: textcat` and `categories` (each with `name` and `description`, at least
2) instead of `entities`; samples rotate through the categories, so the dataset is balanced. Full examples:
`examples/ner/input/prompt.yaml` and `examples/textcat/input/prompt.yaml`.

## Usage

**1. Generate** training data. The rendered prompts of one sample are saved next to the output for review.

```bash
spacylize generate
  --llm-config-path examples/ner/input/llm.yaml
  --prompt-config-path examples/ner/input/prompt.yaml
  --task ner                                        # ner or textcat
  --n-samples 100                                  # optional, default: 10
  --output-path examples/ner/output/train.spacy     # optional, default: data/output.spacy
  --topic-check                                     # optional, default: off; scores each sample against the domain (ollama pull nimble)
  --append                                          # optional, default: off (overwrite, asks first)
```

**2. Validate** the data. This writes a JSON report, a PNG and an HTML report: charts, entity values,
the issues found (duplicates, off-topic texts, label conflicts, unknown labels, leftover markup) and a
list of all documents to filter and select for removal.

```bash
spacylize validate
  --dataset examples/ner/output/train.spacy
  --output-folder examples/ner/output
  --prompt-config examples/ner/input/prompt.yaml    # optional; flags labels that are not configured
  --task ner                                        # optional, default: from the prompt config, or detected
  --open                                            # optional, default: off; opens the HTML report
```

**3. Clean**: remove the documents selected in the report (the report shows this command).

```bash
spacylize clean
  --dataset examples/ner/output/train.spacy
  --remove 3,7,12
  --output examples/ner/output/clean.spacy          # optional, default: overwrite the dataset (asks first)
```

The numbers refer to the validated dataset, so run `validate` again before selecting more.

**4. Split, train and evaluate** (planned, not implemented yet):

```bash
spacylize split
  --input examples/ner/output/train.spacy
  --train examples/ner/output/train_split.spacy     # optional, default: train.spacy
  --dev examples/ner/output/dev_split.spacy         # optional, default: dev.spacy
  --dev-size 0.2                                    # optional, default: 0.2
  --seed 42                                         # optional, default: 42

spacylize train
  --train-data examples/ner/output/train_split.spacy
  --base-model en_core_web_sm                       # optional, default: en_core_web_sm
  --output-model examples/ner/output/model          # optional, default: models/trained_model
  --n-iter 100                                      # optional, default: 100
  --dropout 0.3                                     # optional, default: 0.3

spacylize evaluate
  --model examples/ner/output/model
  --data examples/ner/output/dev_split.spacy
```

The commands are shown with one option per line for readability; write them on one line (or end each
line with `\`, without the comments) to run them.
