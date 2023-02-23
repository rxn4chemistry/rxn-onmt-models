import json
import logging
from pathlib import Path

import click
from rxn.chemutils.multicomponent_smiles import multicomponent_smiles_to_list
from rxn.onmt_utils import Translator
from rxn.utilities.files import iterate_lines_from_file
from rxn.utilities.logging import setup_console_logger

from rxn.onmt_training import __version__
from rxn.onmt_training.forward_predictor import ForwardPredictor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings={"show_default": True})
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the RXN model (forward, retro, etc.)",
)
@click.option(
    "--precursors_file",
    "-p",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the precursors file (one set of precursors per line, in SMILES format)",
)
@click.option(
    "--output_json",
    "-o",
    required=True,
    type=click.Path(writable=True, path_type=Path),
    help="Path where to save the predictions, in .jsonl format (one JSON dict per line)",
)
@click.option("--batch_size", default=64, type=int, help="Batch size")
@click.option(
    "--topn", default=5, type=int, help="Number of retro predictions to make (top-N)"
)
@click.option(
    "--beam_size", default=10, type=int, help="Beam size for retro (> n_best)."
)
@click.option("--no_gpu", is_flag=True, help="Run the training on CPU (slow!)")
def main(
    model: Path,
    precursors_file: Path,
    output_json: Path,
    batch_size: int,
    topn: int,
    beam_size: int,
    no_gpu: bool,
) -> None:
    """Screen multiple lists of precursors with a forward model.

    This will predict one or multiple products for the precursors specified
    on each line of an input file.

    The output .
    The output contains both the raw predictions and the processed / collapsed
    products. It is stored in JSONL file, where each line contains a JSON dictionary
    in the following format (here displayed on multiple lines for readability,
    with topn=3, assuming that the two first predictions collapse):

        {
            "results":[
                {
                    "smiles":[
                        "CCCNC(=O)CC"
                    ],
                    "confidence":0.9999704545228182
                },
                {
                    "smiles":[
                        "CCCN.Cl",
                        "O"
                    ],
                    "confidence":1.642442728161003e-07
                }
            ],
            "raw_results":[
                {
                    "smiles":"CCCNC(=O)CC",
                    "confidence":0.9999581588476526
                },
                {
                    "smiles":"CCCN=C(O)CC",
                    "confidence":1.2295675165578265e-05
                },
                {
                    "smiles":"CCCN~Cl.O",
                    "confidence":1.642442728161003e-07
                }
            ]
        }
    """
    setup_console_logger()

    logger.info(
        f'RXN forward screen "{precursors_file}" -> "{output_json}" with model "{model}". '
    )
    logger.info(f"rxn-onmt-utils version: {__version__}. ")

    translator = Translator.from_model_path(
        model_path=str(model),
        beam_size=beam_size,
        batch_size=batch_size,
        gpu=-1 if no_gpu else 0,
    )
    forward_predictor = ForwardPredictor(model=translator, topn=topn)

    # Convert input to lists of precursors
    precursors_lists = (
        multicomponent_smiles_to_list(line, fragment_bond="~")
        for line in iterate_lines_from_file(precursors_file)
    )

    with open(output_json, "wt") as f:
        for result in forward_predictor.predict(precursors_lists):
            f.write(f"{json.dumps(result)}\n")


if __name__ == "__main__":
    main()
