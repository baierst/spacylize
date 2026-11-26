# Spacylize
**Documentation:** https://baierst.github.io/spacylize

Spacylize is a tool that distills the capabilities of large language models into compact, efficient spaCy models.

**Prerequisites:**
* Python 3.10+
* [uv](https://docs.astral.sh/uv/)

**Installation:**
```bash
uv sync
```

Run the CLI with `uv run spacylize --help`. For development, install the dev tools with `uv sync --group dev`
and run the checks with `uv run black --check .` and `uv run pytest`.

**Local model (Ollama):** the examples use `gemma4` via Ollama by default.

1. Install [Ollama](https://ollama.com) and start it (open the app, or run `ollama serve`).
2. Pull the model: `ollama pull gemma4`

The example `llm.yaml` points spacylize to it:
```yaml
model: ollama_chat/gemma4         # any Ollama model: ollama_chat/<name>
api_base: http://localhost:11434
think: false                      # gemma4 thinks by default; off is much faster
```
To use a cloud provider instead, switch to one of the commented-out models in `llm.yaml` and set its API key
(e.g. `ANTHROPIC_API_KEY`) in your environment or a `.env` file.

# Configuration Guide

Spacylize uses **structured configuration** to make prompt creation quick and easy. Instead of writing complex prompts manually, you specify high-level parameters (entities, domain, tone, etc.) and Jinja2 templates automatically generate optimized prompts.

## Structured Configuration Format

### NER (Named Entity Recognition) Configuration

Create a `prompt.yaml` file with the following structure:

```yaml
task: ner

# Entity definitions
entities:
  - PERSON
  - ORGANIZATION
  - LOCATION
  - DATE

# Generation parameters
domain: "news articles about technology companies"
tone: "professional journalism"
length: "2-3 sentences"
language: "en"  # ISO code: en, de, es, fr, etc.

# Quality controls (optional)
constraints:
  - "Use real company and person names"
  - "Include realistic dates"
  - "Do not repeat entity text"

# Few-shot examples (optional)
examples:
  - text: "[Tim Cook](PERSON) announced that [Apple Inc.](ORGANIZATION) will open a new facility in [Cupertino](LOCATION) by [March 2024](DATE)."
    explanation: "Shows proper inline annotation format"
```

**Required fields:**
- `task`: Must be `"ner"`
- `entities`: List of entity labels (at least 1 required)
- `domain`: Description of the text domain/topic

**Optional fields:**
- `tone`: Writing style (default: "professional")
- `length`: Expected text length (default: "1-2 paragraphs")
- `language`: ISO language code (default: "en")
- `temperature`: Generation randomness 0.0-1.0 (default: provider default; not supported by current Claude models)
- `constraints`: List of additional rules
- `examples`: List of few-shot examples with `text` and optional `explanation`

### TextCat (Text Classification) Configuration

Create a `prompt.yaml` file with the following structure:

```yaml
task: textcat

# Category definitions
categories:
  - name: Electronics
    description: "tech devices, computers, phones, gadgets"
  - name: Clothing
    description: "apparel, shoes, fashion, accessories"
  - name: Home_Goods
    description: "furniture, kitchenware, decor"

# Generation parameters
domain: "e-commerce product descriptions"
tone: "marketing copy"
length: "2-3 sentences"
language: "en"

# Quality controls (optional)
constraints:
  - "Use specific product features"
  - "Include brand names where appropriate"

# Few-shot examples (optional)
examples:
  - text: "This premium wireless keyboard features mechanical switches with customizable RGB lighting."
    category: Electronics
```

**Required fields:**
- `task`: Must be `"textcat"`
- `categories`: List of category definitions with `name` and `description` (at least 2 required)
- `domain`: Description of the text domain/topic

**Optional fields:**
- `tone`: Writing style (default: "professional")
- `length`: Expected text length (default: "2-3 sentences")
- `language`: ISO language code (default: "en")
- `temperature`: Generation randomness 0.0-1.0 (default: provider default; not supported by current Claude models)
- `constraints`: List of additional rules
- `examples`: List of few-shot examples with `text` and `category`

## Verifying Generated Prompts

After running `spacylize generate`, the system automatically saves the rendered prompts to the output directory:

```
output_folder/
├── train.spacy              # Generated training data
├── system_prompt.txt        # Rendered system prompt (for verification)
└── user_prompt.txt          # Rendered user prompt (for verification)
```

You can review `system_prompt.txt` and `user_prompt.txt` to see exactly what prompts were sent to the LLM. This helps with:
- **Debugging**: Understanding why the LLM generated certain outputs
- **Prompt tuning**: Adjusting your structured config to improve results
- **Transparency**: Knowing exactly what instructions the LLM received

# Usage Example:

This example demonstrates how to use `spacylize` to generate training data and train a SpaCy model to identify key attributes from e-commerce product descriptions.

### 1. Create a Prompt Configuration for E-commerce Attributes:

See example: `examples/ner/input/prompt.yaml`

### 2. Generate Training Data:

```bash
spacylize generate --llm-config-path examples/ner/input/llm.yaml --prompt-config-path examples/ner/input/prompt.yaml --n-samples 2000 --output-path examples/ner/output/train.spacy --task ner
```

### 3. Visualize Generated Data:

```bash
spacylize visualize --input-path examples/ner/output/train.spacy --n-samples 5 --port 5002
```

### 4. Validate Data:

```bash
spacylize validate --dataset examples/ner/output/train.spacy --output-folder examples/ner/output --open
```

This writes `train_report.json`, `train_report.png` and an interactive dashboard, `train_report.html`.
The dashboard is a single self-contained file that works offline. It shows:

- the length and label distributions
- an embedding map: similar documents are close together, and hovering shows the text
- quality checks: exact and near duplicates, diversity (distinct-1/2, most repeated openings, top terms)
  and label quality (unique values per label, texts with several labels, documents without a category)

Pass the prompt config used to create the dataset to check its labels against the configured ones:

```bash
spacylize validate --dataset examples/ner/output/train.spacy --output-folder examples/ner/output \
  --prompt-config examples/ner/input/prompt.yaml
```

The report then lists configured labels that are never used and labels in the data that are not
configured. The task is taken from the config, so `--task` is not needed.

With `--topic-check`, `generate` also gives each sample a topic score from a local decision model
(`ollama pull nimble`): how well it fits the configured `domain`. `validate` then lists the off-topic
documents. The check is skipped with a warning if Ollama or the model is not available.

### 5. Split Dataset into Train/Test Sets

```bash
spacylize split --input examples/ner/output/train.spacy --train examples/ner/output/train_split.spacy --dev examples/ner/output/dev_split.spacy --dev-size 0.2 --seed 42
```

### 6. Train a SpaCy Model for Attribute Extraction:

```bash
spacylize train --train-data examples/ner/output/train_split.spacy --base-model en_core_web_sm --output-model examples/ner/output/ecommerce_attribute_model --n-iter 100 --dropout 0.3
```

### 7. Evaluate a Trained SpaCy Model

```bash
spacylize evaluate --model examples/ner/output/ecommerce_attribute_model --data examples/ner/output/dev_split.spacy
```
