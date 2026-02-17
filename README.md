# Spacylize
**Documentation:** https://baierst.github.io/spacylize

Spacylize is a tool that distills the capabilities of large language models into compact, efficient spaCy models.

**Prerequisites:**
* Python 3.8+

**Installation:**
```bash
pip install -e .
```

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
