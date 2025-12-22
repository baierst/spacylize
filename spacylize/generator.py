import yaml
from loguru import logger


class DataGenerator:
    def __init__(
        self, llm_model: str, prompt_config_path: str, n_samples, output_path, task
    ):
        self.llm_model = llm_model

        with open(prompt_config_path, "r") as file:
            self.prompt = yaml.safe_load(file)

        self.n_samples = n_samples
        self.output_path = output_path
        self.task = task

    def run(self):
        logger.warning("DataGenerator is not implemented yet.")
