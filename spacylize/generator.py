import yaml
from spacylize.llm import LLMClient


class DataGenerator:
    def __init__(self, llm_model: str, prompt_config_path: str):
        self.llm = LLMClient(model_name="openai/gpt-4o")

        with open(prompt_config_path, 'r') as file:
            self.prompt = yaml.safe_load(file)

    def generate_sample(self) -> str:
        """Generate a single sample using the LLM based on a provided prompt."""
        return self.llm.completion(self.prompt)

    def generate_data(self, n_samples: int, output_path: str):
        """Generate a set of data samples and save them to the specified path."""
        data = []
        for _ in range(n_samples):
            generated_text = self.generate_sample()
            data.append(generated_text)

        with open(output_path, 'w') as file:
            yaml.dump(data, file)


class NERDataGenerator(DataGenerator):

    def spacy_parse(self):
        pass