"""Data visualization module for exploring SpaCy datasets.

This module provides functionality to visualize SpaCy NER and text classification
datasets using SpaCy's displacy server and custom HTML visualization.
"""

from pathlib import Path
import spacy
from loguru import logger
from spacy.tokens import DocBin
from spacy import displacy


class DataVisualizer:
    """Visualizer for SpaCy datasets.

    Loads a SpaCy dataset and serves an interactive visualization.
    Supports both NER (using displacy) and text classification (custom HTML).

    Attributes:
        input_path: Path to the SpaCy dataset file (.spacy).
        task: The SpaCy task type ('ner' or 'textcat').
        n_samples: Number of samples to visualize.
        port: Port number for the server.
    """

    def __init__(
        self,
        input_path: Path,
        task: str = None,
        n_samples: int = 5,
        port: int = 5002,
    ):
        """Initialize the DataVisualizer.

        Args:
            input_path: Path to the SpaCy data file (.spacy).
            task: Optional task type ('ner' or 'textcat'). Auto-detects if not specified.
            n_samples: Number of samples to visualize. Defaults to 5.
            port: Port to serve the visualization. Defaults to 5002.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the task is invalid or cannot be detected.
        """
        self.input_path = input_path
        self.task = task.lower() if task else None
        self.n_samples = n_samples
        self.port = port

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters and auto-detect task if needed.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the task is invalid or cannot be detected.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        # Auto-detect task if not specified
        if self.task is None:
            self.task = self._detect_task()
            logger.info(f"Auto-detected task type: {self.task}")
        elif self.task not in ["ner", "textcat"]:
            raise ValueError(
                f"Unsupported task: {self.task}. Must be 'ner' or 'textcat'."
            )

    def _detect_task(self) -> str:
        """Auto-detect task type from the dataset.

        Returns:
            str: Detected task type ('ner' or 'textcat').

        Raises:
            ValueError: If task cannot be detected.
        """
        doc_bin = DocBin().from_disk(self.input_path)
        nlp = spacy.blank("en")
        docs = list(doc_bin.get_docs(nlp.vocab))

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
            "Could not detect task type. Dataset appears to have no entities or categories. "
            "Please specify --task explicitly."
        )

    def _load_docbin(self) -> list:
        """Load documents from the SpaCy binary dataset.

        Returns:
            list: First n_samples SpaCy Doc objects from the dataset.
        """
        doc_bin = DocBin().from_disk(self.input_path)
        nlp = spacy.blank("en")
        docs = list(doc_bin.get_docs(nlp.vocab))
        return docs[: self.n_samples]

    def run(self):
        """Run the visualization server.

        Loads the dataset and starts the appropriate visualization server
        based on the task type (NER or textcat).
        """
        docs = self._load_docbin()

        if self.task == "ner":
            logger.info(f"Serving NER visualization at http://localhost:{self.port}")
            displacy.serve(docs, style="ent", port=self.port)
        elif self.task == "textcat":
            self._serve_textcat_visualization(docs)

    def _serve_textcat_visualization(self, docs):
        """Serve custom HTML visualization for text classification.

        Args:
            docs: List of SpaCy Doc objects with category annotations.
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler

        html = [
            "<html><head><style>",
            "body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }",
            "h1 { color: #333; }",
            ".sample { margin: 20px 0; padding: 15px; border: 1px solid #ddd; ",
            "         background: white; border-radius: 5px; }",
            ".sample h3 { margin-top: 0; color: #555; }",
            ".text { font-size: 14px; line-height: 1.6; margin: 10px 0; }",
            ".categories { margin: 10px 0; }",
            ".positive { color: green; font-weight: bold; }",
            ".negative { color: #999; }",
            "</style></head><body>",
            "<h1>Text Classification Samples</h1>",
        ]

        for i, doc in enumerate(docs):
            html.append(f'<div class="sample">')
            html.append(f"<h3>Sample {i+1}</h3>")
            html.append(f'<div class="text"><strong>Text:</strong> {doc.text}</div>')
            html.append(
                '<div class="categories"><strong>Categories:</strong></div><ul>'
            )
            for label, score in sorted(
                doc.cats.items(), key=lambda x: x[1], reverse=True
            ):
                css_class = "positive" if score > 0.5 else "negative"
                html.append(f'<li class="{css_class}">{label}: {score:.2f}</li>')
            html.append("</ul></div>")

        html.append("</body></html>")
        html_content = "".join(html)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html_content.encode())

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        logger.info(f"Serving textcat visualization at http://localhost:{self.port}")
        with HTTPServer(("", self.port), Handler) as httpd:
            httpd.serve_forever()
