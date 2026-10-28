Spacylize Documentation
=======================

Spacylize is a tool that distills the capabilities of large language models into compact, efficient spaCy models.

**Prerequisites:**

* Python 3.10+
* `uv <https://docs.astral.sh/uv/>`_

**Installation:**

.. code-block:: bash

   uv sync

Run the CLI with ``uv run spacylize --help``.

Getting Started
===============

This example demonstrates how to use spacylize to generate training data and train a SpaCy model to identify key attributes from e-commerce product descriptions.

1. Create a Prompt Configuration for E-commerce Attributes
-----------------------------------------------------------

See example: ``examples/ner/prompt.yaml``

2. Generate Training Data
--------------------------

.. code-block:: bash

   spacylize generate --llm-config-path examples/ner/llm.yaml --prompt-config-path examples/ner/prompt.yaml --n-samples 2000 --output-path examples/ner/train.spacy --task ner

3. Visualize Generated Data
----------------------------

.. code-block:: bash

   spacylize visualize --input-path examples/ner/train.spacy --n-samples 5 --port 5002

4. Validate Data
-----------------

.. code-block:: bash

   spacylize validate --dataset examples/ner/train.spacy --output-folder examples/ner

5. Split Dataset into Train/Test Sets
--------------------------------------

.. code-block:: bash

   spacylize split --input examples/ner/train.spacy --train examples/ner/train_split.spacy --dev examples/ner/dev_split.spacy --dev-size 0.2 --seed 42

6. Train a SpaCy Model for Attribute Extraction
------------------------------------------------

.. code-block:: bash

   spacylize train --train-data examples/ner/train_split.spacy --base-model en_core_web_sm --output-model examples/ner/ecommerce_attribute_model --n-iter 100 --dropout 0.3

7. Evaluate a Trained SpaCy Model
----------------------------------

.. code-block:: bash

   spacylize evaluate --model examples/ner/ecommerce_attribute_model --data examples/ner/dev_split.spacy

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   api/cli
   api/generator
   api/validator
   api/visualizer
   api/llm
   api/llm_config
   api/prompt_config
   api/splitter
   api/trainer
   api/evaluator