"""
Functions to launch metrics calculations on forward, retro, or context models.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Type

from rxn.utilities.files import PathLike
from rxn.utilities.logging import setup_console_and_file_logger

from rxn_onmt_utils.rxn_models.forward_or_retro_translation import rxn_translation

from ..scripts.canonicalize_file import canonicalize_file
from ..utils import ensure_directory_exists_and_is_empty
from .context_metrics import ContextMetrics
from .forward_metrics import ForwardMetrics
from .metrics_calculator import MetricsCalculator
from .metrics_files import ContextFiles, ForwardFiles, MetricsFiles, RetroFiles
from .retro_metrics import RetroMetrics
from .tokenize_file import copy_as_detokenized

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_FILES_MAPPING: Dict[str, Type[MetricsFiles]] = {
    "forward": ForwardFiles,
    "context": ContextFiles,
    "retro": RetroFiles,
}
_CALCULATOR_MAPPING: Dict[str, Type[MetricsCalculator]] = {
    "forward": ForwardMetrics,
    "context": ContextMetrics,
    "retro": RetroMetrics,
}


def get_metrics_files(task: str, files_path: PathLike) -> MetricsFiles:
    return _FILES_MAPPING[task](files_path)


def get_metrics_calculator(task: str, files: MetricsFiles) -> MetricsCalculator:
    return _CALCULATOR_MAPPING[task].from_metrics_files(files)


def evaluate_metrics(task: str, files_path: PathLike) -> None:
    logger.info(f"Evaluating the {task} metrics...")
    files = get_metrics_files(task, files_path)
    calculator = get_metrics_calculator(task, files)

    metrics_dict = calculator.get_metrics()

    if files.metrics_file.exists():
        logger.warning(f'Overwriting "{files.metrics_file}"!')

    with open(files.metrics_file, "wt") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f'Evaluating the {task} metrics... Saved to "{files.metrics_file}".')


def run_model_for_metrics(
    task: str,
    model_path: Path,
    src_file: Path,
    tgt_file: Path,
    output_dir: Path,
    n_best: int,
    beam_size: int,
    batch_size: int,
    gpu: bool,
    initialize_logger: bool = False,
) -> None:
    ensure_directory_exists_and_is_empty(output_dir)
    files = get_metrics_files(task, output_dir)

    if initialize_logger:
        setup_console_and_file_logger(files.log_file)

    copy_as_detokenized(src_file, files.gt_src)
    copy_as_detokenized(tgt_file, files.gt_tgt)

    # context prediction
    rxn_translation(
        src_file=files.gt_src,
        tgt_file=files.gt_tgt,
        pred_file=files.predicted,
        model=model_path,
        n_best=n_best,
        beam_size=beam_size,
        batch_size=batch_size,
        gpu=gpu,
    )

    canonicalize_file(
        files.predicted,
        files.predicted_canonical,
        invalid_placeholder="",
        sort_molecules=True,
    )
