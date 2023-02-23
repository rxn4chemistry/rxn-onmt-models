import datetime
import logging
import subprocess
from typing import IO, List, Optional, cast

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def log_file_name_from_time(prefix: Optional[str] = None) -> str:
    """
    Get the name of a log file (typically to create it) from the current
    date and time.

    Returns:
        String for a file name in the format "20221231-1425.log", or
        "{prefix}-20221231-1425.log" if the prefix is specified.
    """
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y%m%d-%H%M")
    if prefix is None:
        return now_formatted + ".log"
    else:
        return prefix + "-" + now_formatted + ".log"


def run_command(command_and_args: List[str]) -> None:
    """
    Run a command, printing its output (stdout and stderr) to the logs.

    Raises:
        RuntimeError: for different errors that may be encountered, and when
            the return code of the executed command is not zero.
    """
    command_str = " ".join(command_and_args)
    command_str_short = f"{command_and_args[0]} [...]"
    logger.info(f"Running command: {command_str}")

    with subprocess.Popen(
        command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        out_stream = cast(IO[bytes], process.stdout)
        try:
            for line in iter(out_stream.readline, b""):
                logger.info(line.decode("utf-8").rstrip())
        except subprocess.CalledProcessError as e:
            msg = f'Error when decoding output of "{command_str_short}"'
            logger.error(msg)
            raise RuntimeError(msg) from e

    return_code = process.returncode

    if return_code == 0:
        logger.info(f'The command "{command_str_short}" ran successfully.')
    else:
        msg = (
            f'The command "{command_str_short}" returned '
            f"code {return_code}. Check the logs for more information."
        )
        logger.error(msg)
        raise RuntimeError(msg)
