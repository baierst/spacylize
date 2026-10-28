import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from spacylize.cli import app


runner = CliRunner()


def test_generate_command_success(tmp_path):
    prompt_config_path = tmp_path / "prompt.yaml"
    prompt_config_path.write_text("dummy: config")

    llm_config_path = tmp_path / "llm.yaml"
    llm_config_path.write_text("dummy: config")

    with patch("spacylize.cli.DataGenerator") as mock_generator:
        instance = mock_generator.return_value

        result = runner.invoke(
            app,
            [
                "generate",
                "--llm-config-path",
                llm_config_path,
                "--prompt-config-path",
                prompt_config_path,
                "--task",
                "ner",
                "--n-samples",
                "10",
                "--output-path",
                str(tmp_path / "out.spacy"),
            ],
        )

        assert result.exit_code == 0
        mock_generator.assert_called_once()
        instance.run.assert_called_once()


def test_generate_command_failure(tmp_path):
    prompt_config_path = tmp_path / "prompt.yaml"
    prompt_config_path.write_text("dummy: config")

    llm_config_path = tmp_path / "llm.yaml"
    llm_config_path.write_text("dummy: config")

    with patch("spacylize.cli.DataGenerator") as mock_generator:
        mock_generator.side_effect = Exception("Boom")

        result = runner.invoke(
            app,
            [
                "generate",
                "--llm-config-path",
                llm_config_path,
                "--prompt-config-path",
                prompt_config_path,
                "--task",
                "ner",
            ],
        )

        assert result.exit_code == 1
        assert "[Error]" in result.output


def test_visualize_command():
    with patch("spacylize.cli.DataVisualizer") as mock_visualizer:
        instance = mock_visualizer.return_value

        result = runner.invoke(
            app,
            [
                "visualize",
                "--input-path",
                "data.spacy",
                "--task",
                "ner",
                "--n-samples",
                "3",
                "--port",
                "5005",
            ],
        )

        assert result.exit_code == 0
        instance.run.assert_called_once()


def test_validate_command(tmp_path):
    # Create a dummy dataset file so Typer passes validation
    dataset_file = tmp_path / "data.spacy"
    dataset_file.touch()

    output_folder = tmp_path / "reports"

    with patch("spacylize.cli.DataValidator") as mock_validator:
        instance = mock_validator.return_value

        result = runner.invoke(
            app,
            [
                "validate",
                "--dataset",
                str(dataset_file),
                "--output-folder",
                str(output_folder),
            ],
        )

        assert result.exit_code == 0

        mock_validator.assert_called_once_with(
            dataset_path=dataset_file,
            output_folder=output_folder,
            task=None,
        )
        instance.run.assert_called_once()


def test_split_command():
    with patch("spacylize.cli.DataSpliter") as mock_splitter:
        instance = mock_splitter.return_value

        result = runner.invoke(
            app,
            [
                "split",
                "--input",
                "data.spacy",
                "--train",
                "train.spacy",
                "--dev",
                "dev.spacy",
                "--dev-size",
                "0.25",
                "--seed",
                "123",
            ],
        )

        assert result.exit_code == 0
        instance.run.assert_called_once()


def test_train_command():
    with patch("spacylize.cli.ModelTrainer") as mock_trainer:
        instance = mock_trainer.return_value

        result = runner.invoke(
            app,
            [
                "train",
                "--train-data",
                "train.spacy",
                "--base-model",
                "en_core_web_sm",
                "--output-model",
                "model/",
                "--n-iter",
                "50",
                "--dropout",
                "0.2",
            ],
        )

        assert result.exit_code == 0
        instance.run.assert_called_once()


def test_evaluate_command():
    with patch("spacylize.cli.ModelEvaluater") as mock_evaluator:
        instance = mock_evaluator.return_value

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "model/",
                "--data",
                "eval.spacy",
            ],
        )

        assert result.exit_code == 0
        instance.run.assert_called_once()
