"""Command-line interface for Spacylize.

This module provides the main CLI application for Spacylize, enabling users
to generate, validate, clean, split, train, and evaluate SpaCy NER datasets
using LLM-powered data generation.
"""

import webbrowser
from pathlib import Path

import typer

from spacylize.evaluator import ModelEvaluator
from spacylize.generator import DataGenerator
from spacylize.splitter import DataSplitter
from spacylize.trainer import ModelTrainer
from spacylize.validator import DataCleaner, DataValidator

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
    topic_check: bool = typer.Option(
        False,
        "--topic-check",
        help="Score each sample against the configured domain with a local "
        "decision model (ollama pull nimble).",
    ),
    append: bool = typer.Option(
        False,
        "--append",
        help="Add the samples to an existing output file instead of overwriting it.",
    ),
) -> None:
    """
    Generates training data using an LLM based on a prompt configuration.
    """
    typer.echo("Generating training data...")
    typer.echo(f"  LLM config path: {llm_config_path}")
    typer.echo(f"  Prompt config path: {prompt_config_path}")
    typer.echo(f"  Number of samples: {n_samples}")
    typer.echo(f"  Output path: {output_path}")
    typer.echo(f"  Task: {task}")

    if output_path.exists() and not append:
        typer.confirm(f"Are you sure you want to overwrite {output_path}?", abort=True)
    try:
        generator = DataGenerator(
            llm_config_path=llm_config_path,
            prompt_config_path=prompt_config_path,
            n_samples=n_samples,
            output_path=output_path,
            task=task,
            topic_check=topic_check,
            append=append,
        )
        generator.run()
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
    open_dashboard: bool = typer.Option(
        False,
        "--open",
        help="Open the HTML dashboard in the browser.",
    ),
    prompt_config_path: Path = typer.Option(
        None,
        "--prompt-config",
        "--prompt-config-path",
        help="Prompt configuration YAML used to create the dataset. If given, the "
        "dataset labels are checked against the configured labels.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """
    CLI entry point for validating a SpaCy dataset.
    """
    try:
        validator = DataValidator(
            dataset_path=dataset,
            output_folder=output_folder,
            task=task,
            prompt_config_path=prompt_config_path,
        )
        validator.run()
        if open_dashboard:
            webbrowser.open(validator.html_path.resolve().as_uri())
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


@app.command(
    name="clean",
    help="Remove the documents selected in the validation report.",
)
def clean(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the SpaCy dataset (.spacy).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    remove: str = typer.Option(
        ...,
        "--remove",
        "-r",
        help="Comma-separated document numbers to remove, e.g. 3,7,12 (as in the report).",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the cleaned dataset. Overwrites the dataset if not given.",
    ),
) -> None:
    """
    CLI entry point for cleaning a SpaCy dataset.
    """
    if output is None:
        typer.confirm(f"Are you sure you want to overwrite {dataset}?", abort=True)
    try:
        ids = {int(i) for i in remove.split(",") if i.strip()}
        cleaner = DataCleaner(dataset, ids, output)
        cleaner.run()
        typer.echo(f"Removed {cleaner.removed} of {cleaner.total} documents.")
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


@app.command(
    name="split",
    help="Split a SpaCy dataset into named sets, e.g. train/dev/test.",
)
def split_dataset(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the SpaCy dataset (.spacy).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_folder: Path = typer.Option(
        ...,
        "--output-folder",
        "-o",
        help="Folder where the sets are written as <name>.spacy.",
        file_okay=False,
        dir_okay=True,
    ),
    splits: list[str] = typer.Option(
        None,
        "--split",
        "-s",
        help="A set as NAME=FRACTION, repeatable, e.g. -s train=0.8 -s dev=0.1 "
        "-s test=0.1. The fractions must sum to 1. Default: train=0.8, dev=0.2.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
) -> None:
    """
    CLI entry point for splitting a SpaCy dataset.
    """
    try:
        fractions = {}
        for item in splits or ["train=0.8", "dev=0.2"]:
            name, sep, fraction = item.partition("=")
            name = name.strip()
            if not sep or not name:
                raise ValueError(f"Invalid split '{item}', expected NAME=FRACTION.")
            if name in fractions:
                raise ValueError(f"Duplicate split name '{name}'.")
            fractions[name] = float(fraction)
        splitter = DataSplitter(input_file, fractions, output_folder, seed)
        for name, path in splitter.run().items():
            typer.echo(f"{name}: {splitter.counts[name]} documents -> {path}")
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


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
) -> None:
    try:
        trainer = ModelTrainer(train_data, base_model, output_model, n_iter, dropout)
        trainer.run()
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


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
) -> None:
    try:
        evaluator = ModelEvaluator(model_path, eval_data)
        evaluator.run()
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
