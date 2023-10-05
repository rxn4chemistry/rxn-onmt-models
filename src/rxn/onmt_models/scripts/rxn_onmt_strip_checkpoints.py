import copy
import logging
from pathlib import Path

import click
from rxn.onmt_utils.strip_model import strip_model
from rxn.utilities.logging import setup_console_logger

from rxn.onmt_models.training_files import ModelFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--model_dir",
    "-m",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory with the model checkpoints to strip.",
)
@click.option(
    "--strip_last_checkpoint",
    is_flag=True,
    help="If specified, the optimizer state will be removed from the last checkpoint as well.",
)
def main(model_dir: Path, strip_last_checkpoint: bool) -> None:
    """Strip the checkpoints (i.e. remove the optimizer state) contained in a model directory.

    By default, it will not remove the optimizer state from the last checkpoint, as
    that one may be needed for finetuning or continued training.

    Also, all the model files that do not incorporate a step number are ignored.

    If you want to strip a single model, use the ``rxn-strip-opennmt-model`` command.
    """
    setup_console_logger()

    model_files = ModelFiles(model_dir)

    all_checkpoints = model_files.get_checkpoints()

    symlink_checkpoints = [p for p in all_checkpoints if p.is_symlink()]
    checkpoints_to_strip = [p for p in all_checkpoints if not p.is_symlink()]

    if symlink_checkpoints:
        print("The following checkpoint(s) are symlinks and will not be stripped:")
        for checkpoint in symlink_checkpoints:
            print(f" - {checkpoint}")

    checkpoints_not_to_strip = copy.deepcopy(symlink_checkpoints)
    if not strip_last_checkpoint:
        checkpoints_not_to_strip.append(checkpoints_to_strip[-1])
        checkpoints_to_strip = checkpoints_to_strip[:-1]

    if checkpoints_to_strip:
        print("The optimizer state will be removed from the following checkpoints:")
        for checkpoint in checkpoints_to_strip:
            print(f" - {checkpoint}")
    else:
        print("No checkpoint to modify.")

    if checkpoints_not_to_strip:
        print("The following checkpoints will not be modified:")
        for checkpoint in checkpoints_not_to_strip:
            print(f" - {checkpoint}")

    confirm = click.confirm("Do you want to proceed?", default=True)

    if not confirm:
        print("Stopping here.")
        return

    for checkpoint in checkpoints_to_strip:
        strip_model(model_in=checkpoint, model_out=checkpoint)


if __name__ == "__main__":
    main()
