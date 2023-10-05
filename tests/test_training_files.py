from pathlib import Path
from typing import Iterable

from rxn.utilities.files import named_temporary_directory, paths_are_identical

from rxn.onmt_models.training_files import ModelFiles


def create_files(directory: Path, files_to_create: Iterable[str]) -> None:
    for filename in files_to_create:
        (directory / filename).touch()


def test_get_checkpoints() -> None:
    with named_temporary_directory() as directory:
        create_files(
            directory,
            [
                "model_ref.pt",
                "model_step_99.pt",
                "model_step_0.pt",
                "model_step_100.pt",
                "model_100000.pt",
            ],
        )

        model_files = ModelFiles(directory)
        checkpoints = model_files.get_checkpoints()

        # check by verifying the names only
        assert [p.name for p in checkpoints] == [
            "model_step_0.pt",
            "model_step_99.pt",
            "model_step_100.pt",
        ]


def test_get_last_checkpoint() -> None:
    with named_temporary_directory() as directory:
        create_files(
            directory,
            [
                "model_ref.pt",
                "model_step_99.pt",
                "model_step_0.pt",
                "model_step_100.pt",
                "model_100000.pt",
            ],
        )

        model_files = ModelFiles(directory)
        last_checkpoint = model_files.get_last_checkpoint()
        assert paths_are_identical(last_checkpoint, directory / "model_step_100.pt")
