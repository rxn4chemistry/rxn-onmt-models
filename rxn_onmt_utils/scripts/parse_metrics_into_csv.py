import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import click
import pandas as pd
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_metric_from_dir(directory: Path) -> Dict[str, Any]:
    """Get the metrics from the metrics.json file in the given directory; a key
    is added for the directory name."""
    with open(directory / "metrics.json", "rt") as f:
        d = json.load(f)
        return {"name": directory.name, **d}


@click.command()
@click.option("--csv", required=True, help="Where to save the csv")
@click.argument("directories", nargs=-1)
def main(csv: str, directories: Tuple[str, ...]) -> None:
    """Parse the metrics from several directories and collect them into a CSV.

    Usage examples:
        - rxn-parse-metrics-into-csv --csv metrics.csv dir1 dir2 dir3
        - rxn-parse-metrics-into-csv --csv metrics.csv dir* other_dir
        - rxn-parse-metrics-into-csv --csv metrics.csv *
    """
    setup_console_logger()

    metrics_dicts = [get_metric_from_dir(Path(directory)) for directory in directories]

    # Note: this flattens the nested dict directly, joining the keys with underscores
    df = pd.json_normalize(metrics_dicts, sep="_")

    df.to_csv(csv, index=False)


if __name__ == "__main__":
    main()
