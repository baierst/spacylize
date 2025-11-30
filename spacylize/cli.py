import typer
from typing import List
from pathlib import Path

from spacylize.generator import DataGenerator
from spacylize.visualizer import DataVisualizer

app = typer.Typer(
    name="spacylize",
    help="""
    Spacylize: LLM-powered data generation and training for SpaCy.
    """,
)


@app.command(
    name="generate",
    help="Generate training data using an LLM.",
)
def generate_data(
    llm: str = typer.Option(
        ...,
        "--llm",
        help="The LLM to use (e.g., gpt-4, mistralai/Mistral-7B-Instruct-v0.2).",
    ),
    prompt_config: Path = typer.Option(
        ...,
        "--prompt-config",
        help="Path to the prompt configuration YAML file.",
    ),
    n_samples: int = typer.Option(
        2000,
        "--n-samples",
        "-n",
        help="Number of samples to generate.",
    ),
    output_path: Path = typer.Option(
        "data/output.spacy",  # Changed default to .spacy
        "--output-path",
        "-o",
        help="Path to save the generated SpaCy data (.spacy format).",
    ),
    task: str = typer.Option(
        ...,
        "--task",
        help="The SpaCy task (e.g., ner, textcat).",
    ),
    labels: List[str] = typer.Option(
        None,
        "--labels",
        help="Labels for the task (e.g., PRODUCT_TYPE,BRAND,...). Comma-separated.",
    ),
):
    """
    Generates training data using an LLM based on a prompt configuration.
    """
    typer.echo("Generating training data...")
    typer.echo(f"  LLM: {llm}")
    typer.echo(f"  Prompt config: {prompt_config}")
    typer.echo(f"  Number of samples: {n_samples}")
    typer.echo(f"  Output path: {output_path}")
    typer.echo(f"  Task: {task}")
    typer.echo(f"  Labels: {labels}")

    try:
        generator = DataGenerator(
            llm_model=llm,
            prompt_config_path=prompt_config,
            n_samples=n_samples,
            output_path=output_path,
            task=task,
            labels=labels,
        )
        generator.run()
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


@app.command(
    name="visualize",
    help="Visualize generated data using SpaCy's displacy.",
)
def visualize_data(
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="Path to the SpaCy data file (.spacy).",
    ),
    task: str = typer.Option(
        ...,
        "--task",
        help="The SpaCy task (e.g., ner, textcat).",
    ),
    n_samples: int = typer.Option(
        5,
        "--n-samples",
        "-n",
        help="Number of samples to visualize.",
    ),
    port: int = typer.Option(
        5002,
        "--port",
        "-p",
        help="Port to serve the visualization.",
    ),
):
    """
    Visualizes generated data using SpaCy's displacy.
    """
    typer.echo("Visualizing generated data...")
    typer.echo(f"  Input path: {input_path}")
    typer.echo(f"  Task: {task}")
    typer.echo(f"  Number of samples: {n_samples}")
    typer.echo(f"  Port: {port}")

    try:
        visualizer = DataVisualizer(
            input_path=input_path,
            task=task,
            n_samples=n_samples,
            port=port,
        )
        visualizer.run()
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


@app.command(
    name="train",
    help="Train a SpaCy pipeline with LLM-generated data.",
)
def train_pipeline(
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        "-t",
        help="Path to the SpaCy training data file (.spacy).",
    ),
    base_model: str = typer.Option(
        "en_core_web_sm",
        "--base-model",
        "-b",
        help="Name of the base SpaCy model to train/fine-tune.",
    ),
    output_model: Path = typer.Option(
        "models/trained_model",
        "--output-model",
        "-o",
        help="Path to save the trained SpaCy pipeline.",
    ),
    n_iter: int = typer.Option(
        100,
        "--n-iter",
        "-n",
        help="Number of training iterations.",
    ),
    dropout: float = typer.Option(
        0.3,
        "--dropout",
        help="Dropout rate during training.",
    )
):
    """
    Trains a SpaCy pipeline using LLM-generated training data.
    """
    typer.echo("Training SpaCy pipeline...")
    typer.echo(f"  Training data: {train_data}")
    typer.echo(f"  Base model: {base_model}")
    typer.echo(f"  Output model: {output_model}")
    typer.echo(f"  Iterations: {n_iter}")
    typer.echo(f"  Dropout: {dropout}")


if __name__ == "__main__":
    app()
