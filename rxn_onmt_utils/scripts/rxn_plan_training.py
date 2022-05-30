#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
from typing import Iterator, List, Union

import click

import rxn_onmt_utils.rxn_models.defaults as defaults
from rxn_onmt_utils.rxn_models.utils import RxnCommand


class Parameter:
    """
    Parameter to be queried to the user, if the command(s) are necessary.
    """

    def __init__(
        self,
        key: str,
        query: str,
        default: Union[int, float],
        commands: RxnCommand,
        optional: bool = True,
    ):
        """
        Args:
            key: parameter name as it is forwarded to the scripts. `
            query: string displayed to the user when querying.
            default: default value for this parameter.
            commands: command(s) that this parameter is needed for.
            optional: if a parameter is optional and its queried value is
                equal to the default, it will not be displayed to the user in
                the command(s) to execute.
        """
        self.key = key
        self.query = query
        self.default = default
        self.type = type(default)
        self.commands = commands
        self.optional = optional


class TrainingPlanner:
    """
    Class that will take the user through the values needed for training models,
    in an interactive manner.
    """

    def __init__(self):
        # All the logic runs directly in the constructor, to avoid the
        # necessity of initially setting all the values to None.
        self.model_task = click.prompt(
            "Please enter the model task", type=click.Choice(["forward", "retro"])
        )

        self._query_about_finetuning()

        self.on_gpu = click.confirm("GPU available?", default=True)
        self.is_multi_task = click.confirm(
            "Do you want to train on multiple data sets (multi-task)?", default=False
        )

        self._get_main_dataset()
        self._maybe_get_additional_dataset_info()

        self.preprocess_seed = click.prompt(
            "Seed for data preprocessing", type=int, default=defaults.SEED
        )

        self.onmt_preprocessed = click.prompt(
            "Where to save the OpenNMT-preprocessed data", type=str
        )
        self.onmt_models = click.prompt("Where to save the OpenNMT models", type=str)

        self._initialize_parameters()
        self._query_parameters()

    def prepare_data_cmd(self) -> Iterator[str]:
        for data_txt, data_dir in zip(self.data_txts, self.data_dirs):
            yield self._prepare_data_cmd(data_txt, data_dir, self.preprocess_seed)

    def preprocess_cmd(self) -> str:
        cmd = (
            "rxn-onmt-preprocess "
            f"--input_dir {self.main_data_dir} "
            f"--output_dir {self.onmt_preprocessed} "
            f"--model_task {self.model_task} "
        )
        if self.is_multi_task:
            for data_dir in self.data_dirs[1:]:
                cmd += f"--additional_data {data_dir} "
        return cmd

    def train_or_finetune_cmd(self) -> str:
        if self.finetuning:
            return self.finetune_cmd()
        else:
            return self.train_cmd()

    def train_cmd(self) -> str:
        cmd = (
            "rxn-onmt-train "
            f"--model_output_dir {self.onmt_models} "
            f"--preprocess_dir {self.onmt_preprocessed} "
        )
        cmd += self._parameters_for_cmd(RxnCommand.T)
        cmd += self._data_weights()
        cmd += self._gpu()
        return cmd

    def finetune_cmd(self) -> str:
        cmd = (
            "rxn-onmt-finetune "
            f"--train_from {self.train_from} "
            f"--model_output_dir {self.onmt_models} "
            f"--preprocess_dir {self.onmt_preprocessed} "
        )
        cmd += self._parameters_for_cmd(RxnCommand.F)
        cmd += self._data_weights()
        cmd += self._gpu()
        return cmd

    def continue_training_cmd(self) -> str:
        cmd = (
            "rxn-onmt-continue-training "
            f"--model_output_dir {self.onmt_models} "
            f"--preprocess_dir {self.onmt_preprocessed} "
        )
        cmd += self._parameters_for_cmd(RxnCommand.C)
        cmd += self._data_weights()
        cmd += self._gpu()
        return cmd

    def _query_about_finetuning(self) -> None:
        self.finetuning = click.confirm(
            "Are you fine-tuning an existing model?", default=False
        )
        if self.finetuning:
            self.needed_commands = [RxnCommand.F, RxnCommand.C]
            self.train_from = click.prompt("Path to the base model", type=str)
        else:
            self.needed_commands = [RxnCommand.T, RxnCommand.C]
            self.train_from = None

    def _get_main_dataset(self) -> None:
        self.main_data_txt = click.prompt("Path to the main data set (TXT)", type=str)
        self.main_data_dir = click.prompt(
            "Where to save the main processed data set", type=str
        )

        # Get all the paths to data
        self.data_txts = [self.main_data_txt]
        self.data_dirs = [self.main_data_dir]
        self.data_weights: List[int] = []

    def _initialize_parameters(self) -> None:
        self.parameters = [
            Parameter("batch_size", "Batch size", defaults.BATCH_SIZE, RxnCommand.TCF),
            Parameter(
                "train_num_steps",
                "Number of training steps",
                100000,
                RxnCommand.TCF,
                optional=False,
            ),
            Parameter(
                "learning_rate", "Learning rate", defaults.LEARNING_RATE, RxnCommand.TF
            ),
            Parameter("dropout", "Dropout", defaults.DROPOUT, RxnCommand.TF),
            Parameter(
                "heads", "Number of transformer heads", defaults.HEADS, RxnCommand.T
            ),
            Parameter("layers", "Number of layers", defaults.LAYERS, RxnCommand.T),
            Parameter("rnn_size", "RNN size", defaults.RNN_SIZE, RxnCommand.T),
            Parameter(
                "transformer_ff",
                "Size of hidden transformer feed-forward",
                defaults.TRANSFORMER_FF,
                RxnCommand.T,
            ),
            Parameter(
                "word_vec_size",
                "Word embedding size",
                defaults.WORD_VEC_SIZE,
                RxnCommand.T,
            ),
            Parameter(
                "warmup_steps",
                "Number of warmup steps",
                defaults.WARMUP_STEPS,
                RxnCommand.TF,
            ),
            Parameter("seed", "Random seed for training", defaults.SEED, RxnCommand.TF),
        ]

    def _query_parameters(self):
        """
        Query the user about the values of all necessary parameters.
        """
        self.param_values = {}
        for p in self.parameters:
            is_needed = any(cmd in p.commands for cmd in self.needed_commands)
            if not is_needed:
                continue

            value = click.prompt(p.query, type=p.type, default=p.default)
            self.param_values[p.key] = value

    def _maybe_get_additional_dataset_info(self) -> None:
        """
        Get the information on additional datasets from the user, if there
        are multiple data sources.
        """
        if not self.is_multi_task:
            return

        number_additional_datasets = click.prompt(
            "Number of additional datasets", type=click.IntRange(min=1)
        )
        for i in range(number_additional_datasets):
            data_txt = click.prompt(
                f"Path to the additional data set (TXT) no {i + 1}", type=str
            )
            data_dir = click.prompt(
                f"Where to save the processed data set no {i + 1}", type=str
            )
            self.data_txts.append(data_txt)
            self.data_dirs.append(data_dir)
        for data_txt in self.data_txts:
            weight = click.prompt(
                f'Training weight for data set in "{data_txt}"',
                type=click.IntRange(min=1),
            )
            self.data_weights.append(weight)

    def _parameters_for_cmd(self, command: RxnCommand) -> str:
        """
        Get the string to append to the command for all the parameters associated
        with a command type.
        """
        to_add = ""
        for p in self.parameters:
            if command not in p.commands:
                continue

            param_value = self.param_values[p.key]
            equal_to_default = param_value == p.default

            if p.optional and equal_to_default:
                continue

            to_add += f"--{p.key} {param_value} "
        return to_add

    @staticmethod
    def _prepare_data_cmd(data_txt: str, data_dir: str, prepare_seed: int) -> str:
        command = f"rxn-prepare-data --input_data {data_txt} --output_dir {data_dir} "
        if prepare_seed != defaults.SEED:
            command += f"--split_seed {prepare_seed} "
        return command

    def _data_weights(self) -> str:
        data_weights = ""
        if self.is_multi_task:
            for weight in self.data_weights:
                data_weights += f"--data_weights {weight} "
        return data_weights

    def _gpu(self) -> str:
        if self.on_gpu:
            return ""
        return "--no_gpu "


@click.command()
def main() -> None:
    """Interactive program to plan the training of RXN OpenNMT models.

    It will ask a user for the values needed for training, and then print all
    the commands to be executed.
    """

    print("Interactive program to plan the training of RXN OpenNMT models.")
    print("NOTE: Please avoid using paths with whitespaces.")

    tp = TrainingPlanner()

    print("Here are the commands to launch a training with RXN:\n")
    print("# 1) Prepare the data (standardization, filtering, etc.)")
    for prepare_cmd in tp.prepare_data_cmd():
        print(prepare_cmd)
    print()
    print(f"# 2) Preprocess the data with OpenNMT\n{tp.preprocess_cmd()}\n")
    print(f"# 3) Train the model\n{tp.train_or_finetune_cmd()}\n")
    print(f"# 4) If necessary: continue training\n{tp.continue_training_cmd()}")


if __name__ == "__main__":
    main()
