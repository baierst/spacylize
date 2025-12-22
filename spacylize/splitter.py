from pathlib import Path

from loguru import logger


class DataSpliter:
    def __init__(
        self,
        input_file: Path,
        train_file: Path,
        dev_file: Path,
        dev_size: float,
        seed: int,
    ):
        self.input_file = input_file
        self.train_file = train_file
        self.dev_file = dev_file
        self.dev_size = dev_size
        self.seed = seed

    def run(self):
        logger.warning("DataSpliter is not yet implemented.")
