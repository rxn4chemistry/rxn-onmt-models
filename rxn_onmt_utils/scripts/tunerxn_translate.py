#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
from pathlib import Path
from typing import Optional

import click

from rxn_onmt_utils.from_tunerxn.translate_onmt import translate_onmt
from rxn_onmt_utils.from_tunerxn.translate_postprocess import translate_postprocess
from rxn_onmt_utils.from_tunerxn.translate_preprocess import translate_preprocess
from rxn_onmt_utils.from_tunerxn.translate_standardize import translate_standardize
from rxn_onmt_utils.from_tunerxn.translate_tokenize import translate_tokenize


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True), required=True)
@click.argument('output_file_path', type=click.Path(), required=True)
@click.option(
    '--model_path', type=str, help='Model path - defaults to the model_path finetuned by IBM'
)
@click.option('--model_task', type=click.Choice(['forward', 'retro']), required=True)
@click.option('--use_gpu', is_flag=True, help='Translate on GPU')
def translate(
    input_file_path: str, output_file_path: str, model_path: Optional[str], model_task: str,
    use_gpu: bool
) -> None:
    """Do an OpenNMT translation - forward or retro"""

    if model_path is None:
        if model_task == 'retro':
            raise SystemExit('No default model is available for retro predictions.')
        model_path = '/models/fwd/rxn-client-ch1_forward_version201127_experiment0.1_bare-lr006_step_100000.pt'

    input_file = Path(input_file_path)
    standardized_file = str(input_file.with_suffix('.standardized.csv'))
    processed_file = str(input_file.with_suffix('.processed.csv'))
    tokenized_file = str(input_file.with_suffix('.tokenized'))
    translation_file = str(input_file.with_suffix('.translated'))

    print(f'Standardize {input_file_path} -> {standardized_file}')
    translate_standardize(
        input_file_path=input_file_path, output_file_path=standardized_file, fragment_bond='~'
    )
    print(f'Preprocess {standardized_file} -> {processed_file}')
    translate_preprocess(
        fragment_bond='~',
        input_file_path=standardized_file,
        output_file_path=processed_file,
        min_reactants=0,
        min_products=0
    )
    translate_tokenize((processed_file, tokenized_file), model_task=model_task)

    print(f'Translating with model {model_path}')
    print(f'Translating... {tokenized_file} -> {translation_file}')
    translate_onmt(
        input_file=tokenized_file,
        output_file=translation_file,
        model=model_path,
        model_task=model_task,
        beam_size=10,
        n_best=5,
        max_length=300,
        use_gpu=use_gpu
    )

    print(f'Post-processing... {input_file_path} and {translation_file} -> {output_file_path}')
    translate_postprocess(
        reactant_input=input_file_path,
        predictions=translation_file,
        combined_output=output_file_path,
        model_task=model_task
    )


if __name__ == '__main__':
    translate()
