# Product Classification Example

This example demonstrates how to use spacylize to generate training data for classifying products into 3 categories: Electronics, Clothing, and Home_Goods.

## Categories

- **Electronics**: Tech devices, computers, phones, gadgets, smart devices
- **Clothing**: Apparel, shoes, fashion items, accessories
- **Home_Goods**: Furniture, kitchenware, decor, household items

## Prerequisites

- Anthropic API key set as `ANTHROPIC_API_KEY` environment variable
- Alternatively, configure `llm.yaml` to use Ollama or another LLM provider

## Generate Training Data

Generate 300 product descriptions with category labels:

```bash
spacylize generate \
  --llm-config-path examples/textcat/llm.yaml \
  --prompt-config-path examples/textcat/prompt.yaml \
  --n-samples 300 \
  --output-path examples/textcat/train.spacy \
  --task textcat
```

## Visualize Sample Data

View generated samples in your browser:

```bash
spacylize visualize \
  --input-path examples/textcat/train.spacy \
  --n-samples 10 \
  --port 5002
```

Then open http://localhost:5002 in your browser.

## Validate the Generated Data

Check the quality and distribution of the generated dataset:

```bash
spacylize validate \
  --dataset examples/textcat/train.spacy \
  --output-folder examples/textcat
```

This will generate:
- `train_report.json` - Statistics about documents, tokens, and category distribution
- `train_report.png` - Visualizations showing:
  - Tokens per document
  - Label distribution (category balance)

## Expected Output Format

Each generated sample follows this delimiter-based format:

```
This stylish leather jacket features a modern slim fit design with
premium quality materials, perfect for both casual and formal occasions.

---
LABEL: Clothing
```

The parser extracts the category label and creates SpaCy training data with document-level category annotations (stored in `doc.cats`).
