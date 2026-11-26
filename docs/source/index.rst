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

**Local model (Ollama):** the examples use ``gemma4`` via Ollama by default.

1. Install `Ollama <https://ollama.com>`_ and start it (open the app, or run ``ollama serve``).
2. Pull the model: ``ollama pull gemma4``

The example ``llm.yaml`` points spacylize to it:

.. code-block:: yaml

   model: ollama_chat/gemma4         # any Ollama model: ollama_chat/<name>
   api_base: http://localhost:11434
   think: false                      # gemma4 thinks by default; off is much faster

To use a cloud provider instead, switch to one of the commented-out models in ``llm.yaml`` and set its API key
(e.g. ``ANTHROPIC_API_KEY``) in your environment or a ``.env`` file.

Getting Started
===============

This example demonstrates how to use spacylize to generate training data and train a SpaCy model to identify key attributes from e-commerce product descriptions.

1. Create a Prompt Configuration for E-commerce Attributes
-----------------------------------------------------------

See example: ``examples/ner/input/prompt.yaml``

2. Generate Training Data
--------------------------

.. code-block:: bash

   spacylize generate \
     --llm-config-path examples/ner/input/llm.yaml \
     --prompt-config-path examples/ner/input/prompt.yaml \
     --n-samples 2000 \
     --output-path examples/ner/output/train.spacy \
     --task ner \
     --topic-check   # optional: score each sample against the domain (needs `ollama pull nimble`)

If the output file exists, ``generate`` asks before overwriting it. To add more samples to an existing
dataset instead, add ``--append``.

3. Validate Data
-----------------

.. code-block:: bash

   spacylize validate --dataset examples/ner/output/train.spacy --output-folder examples/ner/output

4. Split Dataset into Train/Test Sets
--------------------------------------

.. code-block:: bash

   spacylize split --input examples/ner/output/train.spacy --train examples/ner/output/train_split.spacy --dev examples/ner/output/dev_split.spacy --dev-size 0.2 --seed 42

5. Train a SpaCy Model for Attribute Extraction
------------------------------------------------

.. code-block:: bash

   spacylize train --train-data examples/ner/output/train_split.spacy --base-model en_core_web_sm --output-model examples/ner/output/ecommerce_attribute_model --n-iter 100 --dropout 0.3

6. Evaluate a Trained SpaCy Model
----------------------------------

.. code-block:: bash

   spacylize evaluate --model examples/ner/output/ecommerce_attribute_model --data examples/ner/output/dev_split.spacy

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   api/cli
   api/generator
   api/validator
   api/llm
   api/llm_config
   api/prompt_config
   api/splitter
   api/trainer
   api/evaluator