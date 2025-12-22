from pathlib import Path

from loguru import logger


class ModelTrainer:
    def __init__(
        self,
        train_data: Path,
        base_model: str,
        output_model: Path,
        n_iter: int,
        dropout: float,
    ):
        self.train_data = train_data
        self.base_model = base_model
        self.output_model = output_model
        self.n_iter = n_iter
        self.dropout = dropout

    def run(self):
        logger.warning("ModelTrainer is not yet implemented.")
