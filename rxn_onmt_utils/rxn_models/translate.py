import logging
import subprocess
from pathlib import Path
from typing import Union, Optional, List

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def translate(
    model: Union[str, Path], src: Union[str, Path], tgt: Optional[Union[str, Path]],
    output: Union[str,
                  Path], n_best: int, beam_size: int, max_length: int, batch_size: int, gpu: bool
):
    """
    Run translate script.

    This is independent of any chemistry! As such, this does not take care of
    any tokenization either.

    This currently launches a subprocess relying on the OpenNMT binaries.
    In principle, the same could be achieved from Python code directly.
    """
    if not gpu:
        logger.warning('GPU option not set. Only CPUs will be used. The translation may be slow!')

    command: List[str] = [
        'onmt_translate',
        '-model',
        str(model),
        '-src',
        str(src),
        '-output',
        str(output),
        '-log_probs',
        '-n_best',
        str(n_best),
        '-beam_size',
        str(beam_size),
        '-max_length',
        str(max_length),
        '-batch_size',
        str(batch_size),
    ]
    if tgt is not None:
        command.extend(['-tgt', str(tgt)])
    if gpu:
        command.extend(['-gpu', '0'])

    command_str = " ".join(command)
    logger.info(f'Running translation with command: {command_str}')
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        exception_str = f'The command "{command_str}" failed.'
        logger.error(exception_str)
        raise RuntimeError(exception_str) from e
    logger.info('Translation successful.')
