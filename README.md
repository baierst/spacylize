# Spacylize
**Documentation:** https://baierst.github.io/spacylize

Spacylize is a tool that distills the capabilities of large language models into compact, efficient spaCy models.

**Prerequisites:**
* Python 3.8+

**Installation:**
```bash
pip install -e .
```

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
temperature: 0.7  # 0.0-1.0, controls randomness
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
- `temperature`: Generation randomness 0.0-1.0 (default: 0.7)
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
temperature: 0.8
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
- `temperature`: Generation randomness 0.0-1.0 (default: 0.7)
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

## Migration from Old Format

**⚠️ Breaking Change:** Explicit `system` and `user` prompt fields are no longer supported.

If you have an old config like this:

```yaml
system:
  role: system
  content: |
    You are a data generation engine...
user:
  role: user
  content: |
    Generate ONE example...
```

**Convert it to structured format:**

1. Add `task: ner` or `task: textcat` at the top
2. Extract your entities or categories from the old prompts
3. Define `domain` (what the text is about)
4. Optionally set `tone`, `length`, `temperature`, `language`
5. Optionally add `constraints` and `examples`

See the example configs in `examples/ner/prompt.yaml` and `examples/textcat/prompt.yaml` for reference.

# Usage Example:

This example demonstrates how to use `spacylize` to generate training data and train a SpaCy model to identify key attributes from e-commerce product descriptions.

### 1. Create a Prompt Configuration for E-commerce Attributes:

See example: `examples/ecommerce/prompt.yaml`

### 2. Generate Training Data:

```bash
spacylize generate --llm-config-path examples/ner/llm.yaml --prompt-config-path examples/ner/prompt.yaml --n-samples 2000 --output-path examples/ner/train.spacy --task ner
```

### 3. Visualize Generated Data:

```bash
spacylize visualize --input-path examples/ner/train.spacy --n-samples 5 --port 5002
```

### 4. Validate Data:

```bash
spacylize validate --dataset examples/ner/train.spacy --output-folder examples/ner
```

### 5. Split Dataset into Train/Test Sets

```bash
spacylize split --input examples/ner/train.spacy --train examples/ner/train_split.spacy --dev examples/ner/dev_split.spacy --dev-size 0.2 --seed 42
```

### 6. Train a SpaCy Model for Attribute Extraction:

```bash
spacylize train --train-data examples/ner/train_split.spacy --base-model en_core_web_sm --output-model examples/ner/ecommerce_attribute_model --n-iter 100 --dropout 0.3
```

### 7. Evaluate a Trained SpaCy Model

```bash
spacylize evaluate --model examples/ner/ecommerce_attribute_model --data examples/ner/dev_split.spacy
```
