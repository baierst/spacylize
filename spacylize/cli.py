"""Command-line interface for Spacylize.

This module provides the main CLI application for Spacylize, enabling users
to generate, validate, clean, split, train, and evaluate SpaCy NER datasets
using LLM-powered data generation.
"""

import webbrowser
from pathlib import Path
from typing import Annotated

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
    llm_config_path: Annotated[
        Path,
        typer.Option(
            "--llm-config-path",
            help="Path to the LLM configuration YAML file.",
        ),
    ],
    prompt_config_path: Annotated[
        Path,
        typer.Option(
            "--prompt-config-path",
            help="Path to the prompt configuration YAML file.",
        ),
    ],
    task: Annotated[
        str,
        typer.Option(
            "--task",
            help="The SpaCy task (e.g., ner, textcat).",
        ),
    ],
    n_samples: Annotated[
        int,
        typer.Option(
            "--n-samples",
            "-n",
            help="Number of samples to generate.",
        ),
    ] = 10,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="Path to save the generated SpaCy data (.spacy format).",
        ),
    ] = Path("data/output.spacy"),
    topic_check: Annotated[
        bool,
        typer.Option(
            "--topic-check",
            help="Score each sample against the configured domain with a local "
            "decision model (ollama pull nimble).",
        ),
    ] = False,
    append: Annotated[
        bool,
        typer.Option(
            "--append",
            help="Add the samples to an existing output file instead of "
            "overwriting it.",
        ),
    ] = False,
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
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to the SpaCy dataset (.spacy) to validate.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    output_folder: Annotated[
        Path,
        typer.Option(
            "--output-folder",
            "-o",
            help="Folder where the validation report files will be written.",
            file_okay=False,
            dir_okay=True,
        ),
    ],
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="The SpaCy task type (ner or textcat). Auto-detects if not specified.",
        ),
    ] = None,
    open_dashboard: Annotated[
        bool,
        typer.Option(
            "--open",
            help="Open the HTML dashboard in the browser.",
        ),
    ] = False,
    prompt_config_path: Annotated[
        Path | None,
        typer.Option(
            "--prompt-config",
            "--prompt-config-path",
            help="Prompt configuration YAML used to create the dataset. If given, "
            "the dataset labels are checked against the configured labels.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
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
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to the SpaCy dataset (.spacy).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    remove: Annotated[
        str,
        typer.Option(
            "--remove",
            "-r",
            help="Comma-separated document numbers to remove, e.g. 3,7,12 (as in "
            "the report).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the cleaned dataset. Overwrites the dataset if "
            "not given.",
        ),
    ] = None,
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
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the SpaCy dataset (.spacy).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    output_folder: Annotated[
        Path,
        typer.Option(
            "--output-folder",
            "-o",
            help="Folder where the sets are written as <name>.spacy.",
            file_okay=False,
            dir_okay=True,
        ),
    ],
    splits: Annotated[
        list[str] | None,
        typer.Option(
            "--split",
            "-s",
            help="A set as NAME=FRACTION, repeatable, e.g. -s train=0.8 -s dev=0.1 "
            "-s test=0.1. The fractions must sum to 1. Default: train=0.8, dev=0.2.",
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help="Random seed for reproducibility.")
    ] = 42,
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
    train_data: Annotated[
        Path,
        typer.Option(
            "--train-data",
            "-t",
            help="Path to the SpaCy training data file (.spacy).",
        ),
    ],
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            "-b",
            help="Base SpaCy model to train/fine-tune.",
        ),
    ] = "en_core_web_sm",
    output_model: Annotated[
        Path,
        typer.Option(
            "--output-model",
            "-o",
            help="Path to save the trained SpaCy pipeline.",
        ),
    ] = Path("models/trained_model"),
    n_iter: Annotated[
        int, typer.Option("--n-iter", "-n", help="Number of training iterations.")
    ] = 100,
    dropout: Annotated[
        float, typer.Option("--dropout", help="Dropout rate during training.")
    ] = 0.3,
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
    model_path: Annotated[
        Path, typer.Option("--model", "-m", help="Path to the trained SpaCy model.")
    ],
    eval_data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to the evaluation data (.spacy)."),
    ],
) -> None:
    try:
        evaluator = ModelEvaluator(model_path, eval_data)
        evaluator.run()
    except Exception as e:
        typer.echo(f"[Error] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
