from pathlib import Path

from loguru import logger


class DataValidator:
    def __init__(self, dataset: Path):
        """
        Initializes the validator with a SpaCy dataset (.spacy).

        Args:
            dataset (Path): Path to the SpaCy binary dataset.
        """
        self.dataset = dataset

    def run(self):
        logger.warning("ModelEvaluator is not yet implemented.")
