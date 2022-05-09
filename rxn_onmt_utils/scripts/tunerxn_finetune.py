#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import subprocess

import click
from rxn_onmt_utils.model_resize import ModelResizer
from rxn_onmt_utils.from_tunerxn.utils import ModelFiles, OnmtPreprocessedFiles


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--preprocess_dir",
    type=str,
    required=True,
    help="Directory with OpenNMT-preprocessed files",
)
@click.option("--model_output_dir", type=str, required=True, help="Where to save the models")
@click.option("--train_from", type=str, required=True, help="Path to the model to start from")
@click.option("--finetune_num_steps", default=100000)
@click.option("--warmup_steps", default=8000)
@click.option("--batch_size", default=6144)
@click.option("--learning_rate", type=float, default=0.06)
@click.option("--seed", default=42)
@click.option("--use_gpu", is_flag=True, help="Run the training on GPU")
def finetune(
    preprocess_dir: str,
    model_output_dir: str,
    train_from: str,
    finetune_num_steps: int,
    warmup_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    use_gpu: bool,
) -> None:
    """Finetune an OpenNMT model."""

    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    # Extend the vocabulary of the base model based on the training data for
    # finetuning, and save its updated version in the new model directory.
    updated_base_model = str(model_files.model_dir / "updated_base_model.pt")
    resizer = ModelResizer(train_from)
    resizer.extend_vocab(str(onmt_preprocessed_files.vocab_file))
    resizer.save_checkpoint(updated_base_model)

    # yapf: disable
    command_and_args = [
        str(e) for e in [
            'onmt_train',
            '-data', onmt_preprocessed_files.preprocess_prefix,
            '-save_model', model_files.model_prefix,
            '-seed', seed,
            '-save_checkpoint_steps', '5000',
            '-keep_checkpoint', '20',
            '-train_steps', finetune_num_steps,
            '-batch_size', batch_size,
            '-warmup_steps', warmup_steps,
            '-learning_rate', learning_rate,
            '-report_every', '1000',
            '-valid_batch_size', '8',
            '-reset_optim', 'all',
            '-train_from', updated_base_model,
        ]
    ]
    # yapf: enable
    if use_gpu:
        command_and_args.extend(["-gpu_ranks", "0"])

    command_and_args = [str(v) for v in command_and_args]
    print("Running command:", " ".join(command_and_args))
    _ = subprocess.run(command_and_args, check=True)

    print(f"Finetuning successful. Models saved under {str(model_files.model_dir)}")


if __name__ == "__main__":
    finetune()
