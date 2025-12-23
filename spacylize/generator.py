from pathlib import Path
import spacy
from spacy.tokens import DocBin

from spacylize.llm import LLMClient
from spacylize.llm_config import load_llm_config
from spacylize.prompt_config import load_prompt_config


class DataGenerator:
    def __init__(
        self, llm_config_path: Path, prompt_config_path: Path, n_samples, output_path, task
    ):
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

    @staticmethod
    def parse_annotated_text(text: str):
        """
        Parse inline [TEXT](LABEL) annotations into SpaCy entities.
        Returns: clean_text, list of (start, end, label)
        """
        import re

        pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
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

    def run(self):
        nlp = spacy.blank("en")
        doc_bin = DocBin(store_user_data=True)

        for _ in range(self.n_samples):
            result = self.llm_client.generate(
                self.prompt_config.user.content,
                self.prompt_config.system.content,
            )

            clean_text, entities = self.parse_annotated_text(result)

            doc = nlp.make_doc(clean_text)
            spans = []

            for start, end, label in entities:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    spans.append(span)

            doc.ents = spans
            doc_bin.add(doc)

        output_file = self.output_path
        doc_bin.to_disk(output_file)
