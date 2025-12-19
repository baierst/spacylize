from pathlib import Path

from loguru import logger


class ModelEvaluater:
    def __init__(self, model_path: Path, eval_data: Path):
        self.model_path = model_path
        self.eval_data = eval_data

    def run(self):
        logger.warning("ModelEvaluater is not yet implemented.")
