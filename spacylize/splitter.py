"""Dataset splitting module for creating train/dev splits.

This module provides functionality to split SpaCy datasets into training
and development sets for model training and validation.
"""
from pathlib import Path

from loguru import logger


class DataSpliter:
    """Splitter for SpaCy datasets into training and development sets.

    Splits a SpaCy binary dataset into separate training and development
    files using a specified split ratio.

    Attributes:
        input_file: Path to the input SpaCy dataset (.spacy).
        train_file: Path where the training split will be saved.
        dev_file: Path where the development split will be saved.
        dev_size: Fraction of data to allocate to the dev set (0.0-1.0).
        seed: Random seed for reproducible splitting.

    Note:
        This class is not yet fully implemented.
    """

    def __init__(
        self,
        input_file: Path,
        train_file: Path,
        dev_file: Path,
        dev_size: float,
        seed: int,
    ):
        """Initialize the DataSpliter.

        Args:
            input_file: Path to the input SpaCy dataset (.spacy).
            train_file: Output path for the training set.
            dev_file: Output path for the development set.
            dev_size: Fraction of data for the dev set (e.g., 0.2 for 20%).
            seed: Random seed for reproducibility.
        """
        self.input_file = input_file
        self.train_file = train_file
        self.dev_file = dev_file
        self.dev_size = dev_size
        self.seed = seed

    def run(self):
        """Run the dataset splitting process.

        Note:
            This method is not yet implemented.
        """
        logger.warning("DataSpliter is not yet implemented.")