#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED

import numpy as np
import pandas as pd

from rxn_onmt_utils.translator import Translator
from rxn_utilities.file_utilities import load_list_from_file


def translate_onmt(
    input_file: str,
    output_file: str,
    model: str,
    model_task: str,
    use_gpu: bool,
    beam_size: int = 10,
    n_best: int = 5,
    max_length: int = 300
) -> None:
    """Forward reaction translation"""
    if model_task == 'forward':
        input_tag = 'precursors'
        output_tag = 'product'
    elif model_task == 'retro':
        input_tag = 'product'
        output_tag = 'precursors'
    else:
        raise ValueError(f'model_task should be "forward" or "retro" (actual: "{model_task}")')

    translator = Translator.from_model_path(
        model, beam_size=beam_size, max_length=max_length, gpu=(0 if use_gpu else -1)
    )
    input_lines = load_list_from_file(input_file)

    res = translator.translate_multiple_with_scores(input_lines, n_best=n_best)

    dicts = []
    for input_line, output_list in zip(input_lines, res):
        current_dict = {input_tag: input_line}
        for i, translation_result in enumerate(output_list, 1):
            current_dict[f'{output_tag}_{i}'] = translation_result.text
            current_dict[f'confidence_{i}'] = np.exp(translation_result.score)
        dicts.append(current_dict)

    df = pd.DataFrame(dicts)
    df.to_csv(output_file, index=False)
