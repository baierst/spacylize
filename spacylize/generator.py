"""Data generation module for creating SpaCy training datasets using LLMs.

This module provides functionality to generate annotated training data for
SpaCy NER and text classification tasks using Large Language Models.
The generated data is parsed and stored in SpaCy's binary format.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path

import spacy
from spacy.tokens import DocBin

from spacylize.llm import LLMClient
from spacylize.llm_config import load_llm_config
from spacylize.prompt_config import load_prompt_config


class TaskParser(ABC):
    """Base class for parsing LLM-generated annotations."""

    @staticmethod
    @abstractmethod
    def parse(text: str):
        """Parse annotated text and return task-specific data."""
        pass


class DocumentBuilder(ABC):
    """Base class for building SpaCy documents."""

    @staticmethod
    @abstractmethod
    def build(nlp, parsed_data):
        """Build a SpaCy Doc from parsed data."""
        pass


class NERParser(TaskParser):
    """Parser for NER inline annotations: [TEXT](LABEL)"""

    @staticmethod
    def parse(text: str):
        """Parse NER annotations.

        Args:
            text: Annotated text with inline [entity_text](LABEL) markers.

        Returns:
            tuple: (clean_text, entities) where clean_text is the text without
                   annotation markers and entities is a list of (start, end, label)
                   tuples representing entity spans.
        """
        pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        entities = []
        clean_text = ""
        last_end = 0

        for match in pattern.finditer(text):
            start_idx, end_idx = match.span()
            entity_text = match.group(1)
            label = match.group(2)

            clean_text += text[last_end:start_idx]
            start = len(clean_text)

            clean_text += entity_text
            end = len(clean_text)

            entities.append((start, end, label))
            last_end = end_idx

        clean_text += text[last_end:]
        return clean_text, entities


class NERDocumentBuilder(DocumentBuilder):
    """Builds SpaCy docs with entity annotations."""

    @staticmethod
    def build(nlp, parsed_data):
        """Build NER document with entity spans.

        Args:
            nlp: SpaCy language model.
            parsed_data: Tuple of (clean_text, entities).

        Returns:
            SpaCy Doc with entity annotations.
        """
        clean_text, entities = parsed_data
        doc = nlp.make_doc(clean_text)
        spans = []

        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                spans.append(span)

        doc.ents = spans
        return doc


class TextCatParser(TaskParser):
    """Parser for text classification annotations."""

    @staticmethod
    def parse(text: str):
        """Parse textcat annotation format.

        Expected format:
            Text content

            ---
            LABEL: CATEGORY

        Args:
            text: Annotated text with delimiter and label.

        Returns:
            tuple: (clean_text, categories_dict) where categories_dict maps
                   category names to confidence scores.

        Raises:
            ValueError: If the format is invalid.
        """
        # Split on delimiter
        parts = text.split("---")
        if len(parts) != 2:
            raise ValueError("Invalid textcat format: missing '---' delimiter")

        clean_text = parts[0].strip()
        label_section = parts[1].strip()

        # Extract label using regex
        match = re.search(r"LABEL:\s*(\w+)", label_section)
        if not match:
            raise ValueError("Invalid textcat format: missing 'LABEL:' line")

        category = match.group(1)

        # Return single-label classification (exclusive)
        return clean_text, {category: 1.0}


class TextCatDocumentBuilder(DocumentBuilder):
    """Builds SpaCy docs with category annotations."""

    @staticmethod
    def build(nlp, parsed_data):
        """Build textcat document with categories.

        Args:
            nlp: SpaCy language model.
            parsed_data: Tuple of (clean_text, categories_dict).

        Returns:
            SpaCy Doc with category annotations.
        """
        clean_text, categories = parsed_data
        doc = nlp.make_doc(clean_text)
        doc.cats = categories
        return doc


class TaskHandler:
    """Registry for task-specific parsers and builders."""

    _HANDLERS = {
        "ner": (NERParser, NERDocumentBuilder),
        "textcat": (TextCatParser, TextCatDocumentBuilder),
    }

    @classmethod
    def get_handler(cls, task: str):
        """Get parser and builder for a task.

        Args:
            task: The task type (e.g., 'ner', 'textcat').

        Returns:
            tuple: (parser_class, builder_class)

        Raises:
            ValueError: If the task is not supported.
        """
        if task not in cls._HANDLERS:
            supported = ", ".join(cls._HANDLERS.keys())
            raise ValueError(
                f"Unsupported task: '{task}'. Supported tasks: {supported}"
            )
        return cls._HANDLERS[task]

    @classmethod
    def supported_tasks(cls):
        """Get list of supported task types.

        Returns:
            list: List of supported task type strings.
        """
        return list(cls._HANDLERS.keys())


class DataGenerator:
    """Generator for creating SpaCy training data using LLMs."""

    def __init__(
        self,
        llm_config_path: Path,
        prompt_config_path: Path,
        n_samples,
        output_path,
        task,
    ):
        """Initialize the DataGenerator.

        Args:
            llm_config_path: Path to the LLM configuration YAML file.
            prompt_config_path: Path to the prompt configuration YAML file.
            n_samples: Number of training samples to generate.
            output_path: Path where the generated .spacy file will be saved.
            task: The SpaCy task type (e.g., 'ner', 'textcat').
        """
        llm_config = load_llm_config(llm_config_path)

        llm_client = LLMClient(
            model=llm_config.model,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base,
            max_tokens=llm_config.max_tokens,
        )

        prompt_config = load_prompt_config(prompt_config_path)

        self.llm_client = llm_client
        self.prompt_config = prompt_config

        self.n_samples = n_samples
        self.output_path = output_path
        self.task = task

    def run(self):
        """Run the data generation process.

        Generates n_samples of annotated text using the configured LLM,
        parses the annotations using task-specific parsers, and saves
        the results to a SpaCy binary file.
        """
        nlp = spacy.blank("en")
        doc_bin = DocBin(store_user_data=True)

        # Get task-specific parser and builder
        parser_cls, builder_cls = TaskHandler.get_handler(self.task)

        for _ in range(self.n_samples):
            result = self.llm_client.generate(
                self.prompt_config.user.content,
                self.prompt_config.system.content,
            )

            # Parse using task-specific parser
            parsed_data = parser_cls.parse(result)

            # Build document using task-specific builder
            doc = builder_cls.build(nlp, parsed_data)

            doc_bin.add(doc)

        doc_bin.to_disk(self.output_path)
