# Product Classification Example

This example demonstrates how to use spacylize to generate training data for classifying products into 3 categories: Electronics, Clothing, and Home_Goods.

## Categories

- **Electronics**: Tech devices, computers, phones, gadgets
- **Clothing**: Apparel, shoes, fashion, accessories
- **Home_Goods**: Furniture, kitchenware, decor, household items

## Prerequisites

- Ollama running with `gemma4` pulled (`ollama pull gemma4`), see the main README
- For `--topic-check`: `ollama pull nimble`
- Alternatively, switch `llm.yaml` to a cloud model (e.g. Anthropic with `ANTHROPIC_API_KEY`)

## Generate, validate and clean

See the [main README](../../README.md#usage) for all options and their defaults.

```bash
spacylize generate \
  --llm-config-path examples/textcat/input/llm.yaml \
  --prompt-config-path examples/textcat/input/prompt.yaml \
  --task textcat \
  --n-samples 100 \
  --output-path examples/textcat/output/train.spacy \
  --topic-check \
  --append
```

```bash
spacylize validate \
  --dataset examples/textcat/output/train.spacy \
  --output-folder examples/textcat/output \
  --prompt-config examples/textcat/input/prompt.yaml \
  --open
```

The HTML report (`examples/textcat/output/train_report.html`) shows the issues found and all documents. Select the
ones to remove and copy the `spacylize clean` command at the bottom of the report, e.g.:

```bash
spacylize clean \
  --dataset examples/textcat/output/train.spacy \
  --remove 3,7,12
```

## Expected Output Format

Each generated sample follows this delimiter-based format:

```
This stylish leather jacket features a modern slim fit design with
premium quality materials, perfect for both casual and formal occasions.

---
LABEL: Clothing
```

The parser extracts the category label and creates SpaCy training data with document-level category annotations (stored in `doc.cats`).
