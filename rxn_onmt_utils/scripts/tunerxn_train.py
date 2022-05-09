#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import subprocess
from typing import List

import click

from rxn_onmt_utils.from_tunerxn.utils import ModelFiles, OnmtPreprocessedFiles


@click.command(context_settings=dict(show_default=True))
@click.option(
    '--preprocess_dir', type=str, required=True, help='Directory with OpenNMT-preprocessed files'
)
@click.option('--model_output_dir', type=str, required=True, help='Where to save the models')
@click.option('--train_num_steps', default=100000)
@click.option('--warmup_steps', default=8000)
@click.option('--batch_size', default=6144)
@click.option('--learning_rate', type=float, default=2)
@click.option('--layers', default=4)
@click.option('--rnn_size', default=384)
@click.option('--word_vec_size', default=384)
@click.option('--heads', default=8)
@click.option('--dropout', default=0.1)
@click.option('--transformer_ff', default=2048)
@click.option('--seed', default=42)
@click.option('--use_gpu', is_flag=True, help='Run the training on GPU')
def train(
    preprocess_dir: str, model_output_dir: str, train_num_steps: int, warmup_steps: int,
    batch_size: int, learning_rate: float, layers: int, rnn_size: int, word_vec_size: int,
    heads: int, dropout: float, transformer_ff: int, seed: int, use_gpu: bool
) -> None:
    """Train an OpenNMT model."""

    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    # yapf: disable
    command_and_args: List[str] = [
        str(e) for e in [
            'onmt_train',
            '-save_config', model_files.config_file,
            '-data', onmt_preprocessed_files.preprocess_prefix,
            '-save_model', model_files.model_prefix,
            '-seed', seed,
            '-save_checkpoint_steps', '5000',
            '-keep_checkpoint', '20',
            '-train_steps', train_num_steps,
            '-param_init', '0',
            '-param_init_glorot',
            '-max_generator_batches', '32',
            '-batch_size', batch_size,
            '-batch_type', 'tokens',
            '-normalization', 'tokens',
            '-max_grad_norm', '0',
            '-accum_count', '4',
            '-optim', 'adam',
            '-adam_beta1', '0.9',
            '-adam_beta2', '0.998',
            '-decay_method', 'noam',
            '-warmup_steps', warmup_steps,
            '-learning_rate', learning_rate,
            '-label_smoothing', '0.0',
            '-report_every', '1000',
            '-valid_batch_size', '8',
            '-layers', layers,
            '-rnn_size', rnn_size,
            '-word_vec_size', word_vec_size,
            '-encoder_type', 'transformer',
            '-decoder_type', 'transformer',
            '-dropout', dropout,
            '-position_encoding',
            '-share_embeddings',
            '-global_attention', 'general',
            '-global_attention_function', 'softmax',
            '-self_attn_type', 'scaled-dot',
            '-heads', heads,
            '-transformer_ff', transformer_ff,
        ]
    ]
    # yapf: enable
    if use_gpu:
        command_and_args.extend(['-gpu_ranks', '0'])

    # Write config file
    command_and_args = [str(v) for v in command_and_args]
    print('Running command:', ' '.join(command_and_args))
    _ = subprocess.run(command_and_args, check=True)

    # Actual training config file
    command_and_args = ['onmt_train', '-config', str(model_files.config_file)]
    print('Running command:', ' '.join(command_and_args))
    _ = subprocess.run(command_and_args, check=True)

    print(f'Training successful. Models saved under {str(model_files.model_dir)}')


if __name__ == "__main__":
    train()
