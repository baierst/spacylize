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

- Anthropic API key set as `ANTHROPIC_API_KEY` environment variable
- Alternatively, configure `llm.yaml` to use Ollama or another LLM provider

## Generate Training Data

Generate 2000 annotated product descriptions:

```bash
spacylize generate \
  --llm-config-path examples/ner/llm.yaml \
  --prompt-config-path examples/ner/prompt.yaml \
  --n-samples 2000 \
  --output-path examples/ner/train.spacy \
  --task ner
```

## Visualize Sample Data

View annotated samples with entity highlighting in your browser:

```bash
spacylize visualize \
  --input-path examples/ner/train.spacy \
  --n-samples 10 \
  --port 5002
```

Then open http://localhost:5002 in your browser.

## Validate the Generated Data

Check the quality and distribution of entities:

```bash
spacylize validate \
  --dataset examples/ner/train.spacy \
  --output-folder examples/ner
```

This will generate:
- `train_report.json` - Statistics about documents, tokens, and entity distributions
- `train_report.png` - Visualizations showing:
  - Tokens per document
  - Entities per document
  - Entity label distribution
  - Entity length distribution

## Split Dataset

Split the data into training and development sets:

```bash
spacylize split \
  --input examples/ner/train.spacy \
  --train examples/ner/train_split.spacy \
  --dev examples/ner/dev_split.spacy \
  --dev-size 0.2 \
  --seed 42
```

## Train a SpaCy Model

Train a custom SpaCy NER model:

```bash
spacylize train \
  --train-data examples/ner/train_split.spacy \
  --base-model en_core_web_sm \
  --output-model examples/ner/product_ner_model \
  --n-iter 100 \
  --dropout 0.3
```

## Evaluate the Model

Evaluate the trained model on the development set:

```bash
spacylize evaluate \
  --model examples/ner/product_ner_model \
  --data examples/ner/dev_split.spacy
```

## Expected Output Format

Each generated sample follows this inline annotation format:

```
The new [iPhone 15 Pro](MODEL) by [Apple](BRAND) is a premium [Smartphone](PRODUCT_TYPE)
featuring a stunning [6.7 inch](SIZE) display. Available in [Deep Purple](COLOR), this
device comes with [512GB](STORAGE_CAPACITY) of storage and is crafted from
[Titanium](MATERIAL) for exceptional durability.
```

The parser extracts entities and their positions to create SpaCy training data with character-level span annotations.
