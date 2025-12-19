# spacylize (WIP)
Spacylize is a tool that distills the capabilities of large language models into compact, efficient spaCy models.

**Prerequisites:**
* Python 3.8+

**Installation:**
```bash
pip install -e .
```

# Usage Example:

This example demonstrates how to use `spacylize` to generate training data and train a SpaCy model to identify key attributes from e-commerce product descriptions.

### 1. Create a Prompt Configuration for E-commerce Attributes (`examples/ecomerce/promt.yaml`):

```yaml
prompt: "Generate a product description for a {product_type} that includes the following attributes: {attributes_list}. Annotate the following entities using SpaCy's BILUO format: PRODUCT_TYPE, BRAND, MODEL, COLOR, STORAGE_CAPACITY, SIZE, MATERIAL."
entity_types:
  - PRODUCT_TYPE
  - BRAND
  - MODEL
  - COLOR
  - STORAGE_CAPACITY
  - SIZE
  - MATERIAL
examples:
  - product_type: "smartphone"
    attributes_list: "brand: Apple, model: iPhone 15 Pro, color: Space Gray, storage capacity: 256GB"
  - product_type: "running shoes"
    attributes_list: "brand: Nike, model: Air Zoom Pegasus 40, size: US 9, material: mesh"
  - product_type: "coffee maker"
    attributes_list: "brand: Breville, model: Barista Express, material: stainless steel"
```

### 2. Generate Training Data:

```bash
spacylize generate --llm "Mistral-7B-v0.1" --prompt-config examples/ecomerce/promt.yaml --n-samples 2000 --output-path examples/ecomerce/train.spacy --task ner
```

This command uses the `Mistral-7B-v0.1` model to generate 2000 synthetic product descriptions based on the `examples/ecomerce/prompt.yaml` configuration and saves the annotated data in `examples/ecomerce/train.spacy`.

### 3. Visualize Generated Data:

```bash
spacylize visualize --input-path examples/ecomerce/train.spacy --task ner --n-samples 5 --port 5002
```

This will open a web browser showing the annotation of 5 sample product descriptions, allowing you to inspect the generated data and ensure the attributes are being annotated correctly.

### 4. Validate Data:

```bash
spacylize validate --dataset examples/ecommerce/train.spacy
```
This command analyzes the .spacy dataset and produces a quality report.

### 5. Split Dataset into Train/Test Sets

```bash
spacylize split --input examples/ecommerce/train.spacy --train examples/ecommerce/train_split.spacy --dev examples/ecommerce/dev_split.spacy --dev-size 0.2 --seed 42
```
This splits your dataset into training and validation sets. By default, 20% of the data is allocated for the dev set. The split is reproducible with the specified random seed.

### 6. Train a SpaCy Model for Attribute Extraction:

```bash
spacylize train --train-data examples/ecommerce/train_split.spacy --base-model en_core_web_sm --output-model examples/ecommerce/ecommerce_attribute_model --n-iter 100 --dropout 0.3
```
This command trains a new SpaCy model (`models/ecommerce_attribute_model`) using the generated data. It starts with the `en_core_web_sm` base model and trains it for 100 iterations with a dropout rate of 0.3.

### 7. Evaluate a Trained SpaCy Model

```bash
spacylize evaluate --model examples/ecommerce/ecommerce_attribute_model --data examples/ecommerce/dev_split.spacy
```
This evaluates the trained model on the dev set, printing metrics such as precision, recall, and F1-score for all pipeline components.