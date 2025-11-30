# spacylize
Spacylize is a tool that distills the capabilities of large language models into compact, efficient spaCy models.

**Key Features:**

* **LLM-based Data Generation:** Utilizes SpaCy's Hugging Face integration to generate training data for various NLP tasks using powerful LLMs.
  * **Standard SpaCy Training:** Trains standard SpaCy models on the LLM-generated data.
  * **Easy-to-Use CLI:** Provides a simple command-line interface (CLI) to manage the data generation and training processes.
  * **Data Visualization:** Offers a way to visualize the generated data using SpaCy's `displacy` for better understanding and debugging.

**Prerequisites:**
* Python 3.8+

**Installation:**
```bash
pip install .
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
spacylize generate --llm "Mistral-7B-v0.1" --prompt-config examples/ecomerce/promt.yaml --n-samples 2000 --output-path examples/ecomerce/train.spacy --task ner --labels PRODUCT_TYPE,BRAND,MODEL,COLOR,STORAGE_CAPACITY,SIZE,MATERIAL
```

This command uses the `Mistral-7B-v0.1` model to generate 2000 synthetic product descriptions based on the `examples/ecomerce/prompt.yaml` configuration and saves the annotated data in `examples/ecomerce/train.spacy`.

### 3. Visualize Generated Data:

```bash
spacylize visualize --input-path examples/ecomerce/train.spacy --task ner --n-samples 5 --port 5002
```

This will open a web browser showing the annotation of 5 sample product descriptions, allowing you to inspect the generated data and ensure the attributes are being annotated correctly.

### 4. Train a SpaCy Model for Attribute Extraction:

```bash
spacylize train --train-data examples/ecomerce/train.spacy --base-model en_core_web_sm --output-model examples/ecomerce/ecommerce_attribute_model --n-iter 100 --dropout 0.3
```

This command trains a new SpaCy model (`models/ecommerce_attribute_model`) using the generated data. It starts with the `en_core_web_sm` base model and trains it for 100 iterations with a dropout rate of 0.3.
