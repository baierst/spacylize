"""Dataset validation module for SpaCy training data quality assurance.

This module provides functionality to validate SpaCy NER and text classification
datasets, generate quality reports with statistics, and create visualizations
of dataset characteristics.
"""

import json
import statistics
from pathlib import Path
import spacy
from spacy.tokens import DocBin
import matplotlib.pyplot as plt


class DataValidator:
    """Validator for SpaCy datasets that generates quality reports and visualizations.

    Analyzes datasets to compute statistics about document lengths, entity/category
    distributions, and other quality metrics. Supports both NER and text classification
    tasks. Outputs both JSON reports and visualization plots.

    Attributes:
        dataset_path: Path to the SpaCy dataset file (.spacy).
        output_folder: Folder where reports will be saved.
        task: The SpaCy task type ('ner' or 'textcat').
        nlp: Blank SpaCy language model for processing.
        json_path: Path where JSON report will be saved.
        png_path: Path where visualization plots will be saved.
    """

    def __init__(self, dataset_path: str, output_folder: str, task: str = None):
        """Initialize the DataValidator.

        Args:
            dataset_path: Path to the SpaCy dataset file (.spacy) to validate.
            output_folder: Directory where validation reports will be saved.
            task: Optional task type ('ner' or 'textcat'). Auto-detects if not specified.

        Raises:
            ValueError: If the task is invalid or cannot be detected.
        """
        self.dataset_path = Path(dataset_path)
        self.output_folder = Path(output_folder)
        self.nlp = spacy.blank("en")

        # Auto-detect task if not specified
        if task is None:
            docs = self._load_docs()
            self.task = self._detect_task(docs)
            from loguru import logger
            logger.info(f"Auto-detected task type: {self.task}")
        else:
            self.task = task.lower()
            if self.task not in ["ner", "textcat"]:
                raise ValueError(f"Unsupported task: {self.task}. Must be 'ner' or 'textcat'.")

        dataset_name = self.dataset_path.stem
        self.json_path = self.output_folder / f"{dataset_name}_report.json"
        self.png_path = self.output_folder / f"{dataset_name}_report.png"

    def run(self):
        """Run the validation process and generate reports.

        Analyzes the dataset using task-specific validation logic,
        computes statistics, and generates both a JSON report and
        visualization plots.

        Raises:
            ValueError: If the task type is not supported.
        """
        docs = self._load_docs()

        if self.task == "ner":
            self._validate_ner(docs)
        elif self.task == "textcat":
            self._validate_textcat(docs)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _validate_ner(self, docs):
        """Validate NER dataset and generate reports.

        Args:
            docs: List of SpaCy Doc objects with NER annotations.
        """
        doc_lengths = []
        ents_per_doc = []
        entity_lengths = []
        entity_label_counts = {}

        total_tokens = 0
        total_entities = 0

        for doc in docs:
            doc_len = len(doc)
            num_ents = len(doc.ents)

            doc_lengths.append(doc_len)
            ents_per_doc.append(num_ents)

            total_tokens += doc_len
            total_entities += num_ents

            for ent in doc.ents:
                entity_lengths.append(len(ent))
                entity_label_counts[ent.label_] = (
                    entity_label_counts.get(ent.label_, 0) + 1
                )

        report = {
            "dataset": {
                "path": str(self.dataset_path),
                "num_documents": len(docs),
                "num_tokens": total_tokens,
                "num_entities": total_entities,
            },
            "documents": {
                "tokens_per_doc": self._summary(doc_lengths),
                "entities_per_doc": self._summary(ents_per_doc),
            },
            "entities": {
                "total": total_entities,
                "by_label": dict(sorted(entity_label_counts.items())),
                "entity_length_tokens": self._summary(entity_lengths),
            },
        }

        self._write_json(report)
        self._write_plots(
            doc_lengths,
            ents_per_doc,
            entity_label_counts,
            entity_lengths,
        )

    def _validate_textcat(self, docs):
        """Validate text classification dataset and generate reports.

        Args:
            docs: List of SpaCy Doc objects with category annotations.
        """
        doc_lengths = []
        label_counts = {}

        for doc in docs:
            doc_lengths.append(len(doc))

            # Count positive labels (score > 0.5)
            for label, score in doc.cats.items():
                if score > 0.5:
                    label_counts[label] = label_counts.get(label, 0) + 1

        report = {
            "dataset": {
                "path": str(self.dataset_path),
                "num_documents": len(docs),
                "num_tokens": sum(doc_lengths),
            },
            "documents": {
                "tokens_per_doc": self._summary(doc_lengths),
            },
            "labels": {
                "total_unique": len(label_counts),
                "distribution": dict(sorted(label_counts.items())),
            },
        }

        self._write_json(report)
        self._write_textcat_plots(doc_lengths, label_counts)

    def _load_docs(self):
        """Load documents from the SpaCy binary dataset.

        Returns:
            list: List of SpaCy Doc objects from the dataset.
        """
        doc_bin = DocBin().from_disk(self.dataset_path)
        return list(doc_bin.get_docs(self.nlp.vocab))

    def _detect_task(self, docs) -> str:
        """Auto-detect task type from the dataset.

        Args:
            docs: List of SpaCy Doc objects.

        Returns:
            str: Detected task type ('ner' or 'textcat').

        Raises:
            ValueError: If task cannot be detected.
        """
        if not docs:
            raise ValueError("Dataset is empty, cannot detect task type")

        # Check first document
        first_doc = docs[0]

        # Check for NER entities
        if first_doc.ents:
            return "ner"

        # Check for text classification categories
        if first_doc.cats:
            return "textcat"

        raise ValueError(
            "Could not detect task type. Dataset appears to have no entities or categories."
        )

    def _summary(self, values):
        """Compute summary statistics for a list of values.

        Args:
            values: List of numeric values.

        Returns:
            dict: Dictionary with 'min', 'max', and 'mean' keys.
        """
        if not values:
            return {"min": 0, "max": 0, "mean": 0}

        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 2),
        }

    def _write_json(self, report):
        """Write the validation report to a JSON file.

        Args:
            report: Dictionary containing validation statistics.
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("w", encoding="utf8") as f:
            json.dump(report, f, indent=2)

    def _write_plots(
        self,
        doc_lengths,
        ents_per_doc,
        entity_label_counts,
        entity_lengths,
    ):
        """Generate and save visualization plots for the dataset.

        Args:
            doc_lengths: List of document lengths in tokens.
            ents_per_doc: List of entity counts per document.
            entity_label_counts: Dictionary mapping entity labels to counts.
            entity_lengths: List of entity lengths in tokens.
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("NER Dataset Validation Report", fontsize=16)

        # Tokens per document
        axes[0, 0].hist(doc_lengths, bins=30)
        axes[0, 0].set_title("Tokens per Document")
        axes[0, 0].set_xlabel("Tokens")
        axes[0, 0].set_ylabel("Documents")

        # Entities per document
        axes[0, 1].hist(ents_per_doc, bins=30)
        axes[0, 1].set_title("Entities per Document")
        axes[0, 1].set_xlabel("Entities")
        axes[0, 1].set_ylabel("Documents")

        # Entity label distribution
        labels = list(entity_label_counts.keys())
        counts = list(entity_label_counts.values())

        axes[1, 0].bar(labels, counts)
        axes[1, 0].set_title("Entity Label Distribution")
        axes[1, 0].set_xlabel("Entity Label")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Entity length distribution
        if entity_lengths:
            bins = range(1, max(entity_lengths) + 2)
        else:
            bins = [1]

        axes[1, 1].hist(entity_lengths, bins=bins)
        axes[1, 1].set_title("Entity Length (Tokens)")
        axes[1, 1].set_xlabel("Tokens")
        axes[1, 1].set_ylabel("Entities")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.png_path)
        plt.close(fig)

    def _write_textcat_plots(self, doc_lengths, label_counts):
        """Generate and save visualization plots for text classification dataset.

        Args:
            doc_lengths: List of document lengths in tokens.
            label_counts: Dictionary mapping category labels to counts.
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Text Classification Dataset Validation", fontsize=16)

        # Tokens per document
        axes[0].hist(doc_lengths, bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title("Tokens per Document")
        axes[0].set_xlabel("Tokens")
        axes[0].set_ylabel("Count")

        # Label distribution
        if label_counts:
            labels = list(label_counts.keys())
            counts = list(label_counts.values())
            axes[1].bar(labels, counts, color='lightcoral', edgecolor='black')
            axes[1].set_title("Label Distribution")
            axes[1].set_xlabel("Category")
            axes[1].set_ylabel("Count")
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.png_path)
        plt.close(fig)