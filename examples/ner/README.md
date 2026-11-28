# E-commerce Product NER Example

This example demonstrates how to use spacylize to generate training data for Named Entity Recognition (NER) to extract product attributes from e-commerce product descriptions.

## Entity Types

This example extracts the following entities:
- **PRODUCT_TYPE**: Type of product (e.g., Smartphone, Laptop, Smartwatch)
- **BRAND**: Brand name (e.g., Apple, Samsung, Google)
- **MODEL**: Product model (e.g., iPhone 15, Galaxy S24)
- **COLOR**: Product color (e.g., Space Gray, Midnight Blue)
- **STORAGE_CAPACITY**: Storage size (e.g., 256GB, 1TB)
- **SIZE**: Product dimensions (e.g., 6.1 inches, 14-inch)
- **MATERIAL**: Product material (e.g., Aluminum, Titanium)

## Prerequisites

- Ollama running with `gemma4` pulled (`ollama pull gemma4`), see the main README
- For `--topic-check`: `ollama pull nimble`
- Alternatively, switch `llm.yaml` to a cloud model (e.g. Anthropic with `ANTHROPIC_API_KEY`)

## Generate, validate and clean

See the [main README](../../README.md#usage) for all options and their defaults.

```bash
spacylize generate \
  --llm-config-path examples/ner/input/llm.yaml \
  --prompt-config-path examples/ner/input/prompt.yaml \
  --task ner \
  --n-samples 100 \
  --output-path examples/ner/output/data.spacy \
  --topic-check \
  --append
```

```bash
spacylize validate \
  --dataset examples/ner/output/data.spacy \
  --output-folder examples/ner/output \
  --prompt-config examples/ner/input/prompt.yaml \
  --open
```

The HTML report (`examples/ner/output/train_report.html`) shows the issues found and all documents. Select the
ones to remove and copy the `spacylize clean` command at the bottom of the report, e.g.:

```bash
spacylize clean \
  --dataset examples/ner/output/data.spacy \
  --remove 3,7,12
```

## Expected Output Format

Each sample asks for a random subset of the entity types and uses this inline annotation format:

```
The new [iPhone 15 Pro](MODEL) by [Apple](BRAND) features a stunning [6.7 inch](SIZE) display
and comes in [Deep Purple](COLOR).
```

The parser extracts entities and their positions to create SpaCy training data with character-level span annotations.
