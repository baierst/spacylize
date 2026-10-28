"""Command-line interface for Spacylize.

This module provides the main CLI application for Spacylize, enabling users
to generate, validate, visualize, split, train, and evaluate SpaCy NER datasets
using LLM-powered data generation.
"""

import typer
from pathlib import Path

from spacylize.evaluator import ModelEvaluater
from spacylize.generator import DataGenerator
from spacylize.splitter import DataSpliter
from spacylize.trainer import ModelTrainer
from spacylize.validator import DataValidator
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
    llm_config_path: Path = typer.Option(
        ...,
        "--llm-config-path",
        help="Path to the LLM configuration YAML file.",
    ),
    prompt_config_path: Path = typer.Option(
        ...,
        "--prompt-config-path",
        help="Path to the prompt configuration YAML file.",
    ),
    n_samples: int = typer.Option(
        10,
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
):
    """
    Generates training data using an LLM based on a prompt configuration.
    """
    typer.echo("Generating training data...")
    typer.echo(f"  LLM config path: {llm_config_path}")
    typer.echo(f"  Prompt config path: {prompt_config_path}")
    typer.echo(f"  Number of samples: {n_samples}")
    typer.echo(f"  Output path: {output_path}")
    typer.echo(f"  Task: {task}")

    try:
        generator = DataGenerator(
            llm_config_path=llm_config_path,
            prompt_config_path=prompt_config_path,
            n_samples=n_samples,
            output_path=output_path,
            task=task,
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
        None,
        "--task",
        help="The SpaCy task (e.g., ner, textcat). Auto-detects if not specified.",
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
    name="validate",
    help="Validate a SpaCy dataset and produce a quality report.",
)
def validate_dataset(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the SpaCy dataset (.spacy) to validate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_folder: Path = typer.Option(
        ...,
        "--output-folder",
        "-o",
        help="Folder where the validation report files will be written.",
        file_okay=False,
        dir_okay=True,
    ),
    task: str = typer.Option(
        None,
        "--task",
        "-t",
        help="The SpaCy task type (ner or textcat). Auto-detects if not specified.",
    ),
):
    """
    CLI entry point for validating a SpaCy dataset.
    """
    validator = DataValidator(
        dataset_path=dataset,
        output_folder=output_folder,
        task=task,
    )
    validator.run()


@app.command(
    name="split",
    help="Split a SpaCy binary dataset into training and dev sets.",
)
def split_dataset(
    input_file: Path = typer.Option(
        ..., "--input", "-i", help="Path to the SpaCy dataset (.spacy)."
    ),
    train_file: Path = typer.Option(
        "train.spacy", "--train", "-t", help="Output path for training set."
    ),
    dev_file: Path = typer.Option(
        "dev.spacy", "--dev", "-d", help="Output path for dev set."
    ),
    dev_size: float = typer.Option(
        0.2, "--dev-size", help="Fraction of data for dev set."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
):
    """
    CLI entry point for splitting a SpaCy dataset.
    """
    splitter = DataSpliter(input_file, train_file, dev_file, dev_size, seed)
    splitter.run()


@app.command(
    name="train",
    help="Train a SpaCy pipeline with LLM-generated data.",
)
def train_pipeline(
    train_data: Path = typer.Option(
        ..., "--train-data", "-t", help="Path to the SpaCy training data file (.spacy)."
    ),
    base_model: str = typer.Option(
        "en_core_web_sm",
        "--base-model",
        "-b",
        help="Base SpaCy model to train/fine-tune.",
    ),
    output_model: Path = typer.Option(
        "models/trained_model",
        "--output-model",
        "-o",
        help="Path to save the trained SpaCy pipeline.",
    ),
    n_iter: int = typer.Option(
        100, "--n-iter", "-n", help="Number of training iterations."
    ),
    dropout: float = typer.Option(
        0.3, "--dropout", help="Dropout rate during training."
    ),
):
    trainer = ModelTrainer(train_data, base_model, output_model, n_iter, dropout)
    trainer.run()


@app.command(
    name="evaluate",
    help="Evaluate a trained SpaCy model.",
)
def evaluate_model(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to the trained SpaCy model."
    ),
    eval_data: Path = typer.Option(
        ..., "--data", "-d", help="Path to the evaluation data (.spacy)."
    ),
):
    evaluator = ModelEvaluater(model_path, eval_data)
    evaluator.run()


if __name__ == "__main__":
    app()
