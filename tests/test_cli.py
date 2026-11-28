from unittest.mock import patch

from typer.testing import CliRunner

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
            prompt_config_path=None,
        )
        instance.run.assert_called_once()


def test_split_command(tmp_path):
    dataset_file = tmp_path / "data.spacy"
    dataset_file.touch()
    output_folder = tmp_path / "splits"

    with patch("spacylize.cli.DataSplitter") as mock_split:
        result = runner.invoke(
            app,
            [
                "split",
                "--input",
                str(dataset_file),
                "--output-folder",
                str(output_folder),
                "--split",
                "train=0.8",
                "--split",
                "dev=0.1",
                "--split",
                "test=0.1",
                "--seed",
                "123",
            ],
        )

        assert result.exit_code == 0
        mock_split.assert_called_once_with(
            dataset_file,
            {"train": 0.8, "dev": 0.1, "test": 0.1},
            output_folder,
            123,
        )
        mock_split.return_value.run.assert_called_once()


def test_split_command_default_splits(tmp_path):
    dataset_file = tmp_path / "data.spacy"
    dataset_file.touch()

    with patch("spacylize.cli.DataSplitter") as mock_split:
        result = runner.invoke(
            app, ["split", "-i", str(dataset_file), "-o", str(tmp_path / "splits")]
        )

        assert result.exit_code == 0
        assert mock_split.call_args.args[1] == {"train": 0.8, "dev": 0.2}


def test_split_command_invalid_split(tmp_path):
    dataset_file = tmp_path / "data.spacy"
    dataset_file.touch()

    with patch("spacylize.cli.DataSplitter") as mock_split:
        result = runner.invoke(
            app,
            ["split", "-i", str(dataset_file), "-o", str(tmp_path), "-s", "train"],
        )

        assert result.exit_code == 1
        assert "[Error]" in result.output
        mock_split.assert_not_called()


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
    with patch("spacylize.cli.ModelEvaluator") as mock_evaluator:
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


def test_evaluate_command_error_exits_cleanly():
    with patch("spacylize.cli.ModelEvaluator") as mock_evaluator:
        mock_evaluator.return_value.run.side_effect = RuntimeError("boom")

        result = runner.invoke(
            app,
            ["evaluate", "--model", "model/", "--data", "eval.spacy"],
        )

        assert result.exit_code == 1
        assert "[Error] boom" in result.output


def test_clean_command(tmp_path):
    dataset_file = tmp_path / "data.spacy"
    dataset_file.touch()

    with patch("spacylize.cli.DataCleaner") as mock_clean:
        mock_clean.return_value.removed = 2
        mock_clean.return_value.total = 10
        # Overwriting the dataset asks for confirmation; "n" aborts
        args = ["clean", "--dataset", str(dataset_file), "--remove", "1,3"]
        result = runner.invoke(app, args, input="n\n")
        assert result.exit_code == 1
        mock_clean.assert_not_called()

        result = runner.invoke(app, args, input="y\n")

    assert result.exit_code == 0
    assert "Removed 2 of 10 documents" in result.stdout
    mock_clean.assert_called_once_with(dataset_file, {1, 3}, None)
    mock_clean.return_value.run.assert_called_once()


def test_generate_asks_before_overwriting(tmp_path):
    for name in ("prompt.yaml", "llm.yaml", "out.spacy"):
        (tmp_path / name).write_text("x")
    args = [
        "generate",
        "--llm-config-path",
        str(tmp_path / "llm.yaml"),
        "--prompt-config-path",
        str(tmp_path / "prompt.yaml"),
        "--task",
        "ner",
        "--output-path",
        str(tmp_path / "out.spacy"),
    ]

    with patch("spacylize.cli.DataGenerator") as mock_generator:
        # The output exists: "n" aborts, --append does not ask
        assert runner.invoke(app, args, input="n\n").exit_code == 1
        mock_generator.assert_not_called()

        assert runner.invoke(app, args + ["--append"]).exit_code == 0
        assert mock_generator.call_args.kwargs["append"] is True
