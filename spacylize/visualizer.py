"""Data visualization module for exploring SpaCy datasets.

This module provides functionality to visualize SpaCy NER datasets using
SpaCy's displacy server, enabling interactive exploration of annotated entities.
"""
from pathlib import Path
import spacy
from loguru import logger
from spacy.tokens import DocBin
from spacy import displacy


class DataVisualizer:
    """Visualizer for SpaCy datasets using displacy server.

    Loads a SpaCy dataset and serves an interactive visualization using
    SpaCy's built-in displacy server for exploring entity annotations.

    Attributes:
        input_path: Path to the SpaCy dataset file (.spacy).
        task: The SpaCy task type (currently only 'ner' is supported).
        n_samples: Number of samples to visualize.
        port: Port number for the displacy server.
    """

    def __init__(
        self,
        input_path: Path,
        task: str,
        n_samples: int = 5,
        port: int = 5002,
    ):
        """Initialize the DataVisualizer.

        Args:
            input_path: Path to the SpaCy data file (.spacy).
            task: The SpaCy task type (e.g., 'ner').
            n_samples: Number of samples to visualize. Defaults to 5.
            port: Port to serve the visualization. Defaults to 5002.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the task is not supported.
        """
        self.input_path = input_path
        self.task = task.lower()
        self.n_samples = n_samples
        self.port = port

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the task is not 'ner'.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if self.task != "ner":
            raise ValueError("Currently only 'ner' task is supported.")

    def _load_docbin(self) -> list:
        """Load documents from the SpaCy binary dataset.

        Returns:
            list: First n_samples SpaCy Doc objects from the dataset.
        """
        doc_bin = DocBin().from_disk(self.input_path)
        nlp = spacy.blank("en")
        docs = list(doc_bin.get_docs(nlp.vocab))
        return docs[: self.n_samples]

    def run(self):
        """Run the visualization server.

        Loads the dataset and starts the displacy server to visualize
        entity annotations in a web browser.
        """
        docs = self._load_docbin()
        displacy.serve(docs, style="ent", port=self.port)