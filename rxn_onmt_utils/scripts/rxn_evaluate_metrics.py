import click
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils.rxn_models.run_metrics import evaluate_metrics


@click.command(context_settings={"show_default": True})
@click.option(
    "--task", required=True, type=click.Choice(["forward", "retro", "context"])
)
@click.option(
    "--results_dir", required=True, help="Where the retro predictions are stored"
)
def main(task: str, results_dir: str) -> None:
    """Evaluate the metrics (the metrics must have been generated already!)"""

    setup_console_logger()

    evaluate_metrics(task, results_dir)


if __name__ == "__main__":
    main()
