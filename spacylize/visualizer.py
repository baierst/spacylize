from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy import displacy


class DataVisualizer:
    def __init__(
        self,
        input_path: Path,
        task: str,
        n_samples: int = 5,
        port: int = 5002,
    ):
        self.input_path = input_path
        self.task = task.lower()
        self.n_samples = n_samples
        self.port = port

        self._validate_inputs()

    def _validate_inputs(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if self.task != "ner":
            raise ValueError("Currently only 'ner' task is supported.")

    def _load_docbin(self) -> list:
        doc_bin = DocBin().from_disk(self.input_path)
        nlp = spacy.blank("en")
        docs = list(doc_bin.get_docs(nlp.vocab))
        return docs[: self.n_samples]

    def run(self):
        print(f"[spacylize] Loading {self.n_samples} samples from {self.input_path}...")
        docs = self._load_docbin()
        print(f"[spacylize] Launching displacy server on port {self.port}")
        displacy.serve(docs, style="ent", port=self.port)
