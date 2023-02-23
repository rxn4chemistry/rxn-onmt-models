from typing import Iterator, List, Optional, Union

import click
from attr import define
from rxn.onmt_utils.train_command import RxnCommand

import rxn.onmt_models.defaults as defaults

_CONTEXT_DATA_BATCH_SIZE = 8


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


@define
class ContextOptions:
    tagging_batch_size: int


@define
class AugmentOptions:
    number_augmentations: int


@define
class DatasetOptions:
    txt_path: str
    processed_path: str
    weight: int
    augment: Optional[AugmentOptions]


class TrainingPlanner:
    """
    Class that will take the user through the values needed for training models,
    in an interactive manner.
    """

    def __init__(self) -> None:
        # All the logic runs directly in the constructor, to avoid the
        # necessity of initially setting all the values to None.
        self.model_task = click.prompt(
            "Please enter the model task",
            type=click.Choice(["forward", "retro", "context"]),
        )

        self._query_about_finetuning()

        self.on_gpu = click.confirm("GPU available?", default=True)

        self.datasets = self._get_datasets()

        self.preprocess_seed = click.prompt(
            "Seed for data preprocessing", type=int, default=defaults.SEED
        )

        self.context_options = self._maybe_get_context_options()

        self.onmt_preprocessed = click.prompt(
            "Where to save the OpenNMT-preprocessed data", type=str
        )
        self.onmt_models = click.prompt("Where to save the OpenNMT models", type=str)

        self._initialize_parameters()
        self._query_parameters()

    def prepare_data_cmd(self) -> Iterator[str]:
        for dataset in self.datasets:
            yield self._prepare_data_cmd(dataset, self.preprocess_seed)

    def prepare_context_data_cmd(self) -> Iterator[str]:
        for dataset in self.datasets:
            yield self._prepare_context_data_cmd(dataset.processed_path)

    def augment_data_cmd(self) -> Iterator[str]:
        for dataset in self.datasets:
            cmd = self._augment_cmd(dataset)
            if cmd is not None:
                yield cmd

    def preprocess_cmd(self) -> str:
        cmd = (
            "rxn-onmt-preprocess "
            f"--input_dir {self.datasets[0].processed_path} "
            f"--output_dir {self.onmt_preprocessed} "
            f"--model_task {self.model_task} "
        )
        for dataset in self.datasets[1:]:
            cmd += f"--additional_data {dataset.processed_path} "
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

    def _query_parameters(self) -> None:
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

    def _get_datasets(self) -> List[DatasetOptions]:
        """
        Get the information on datasets from the user.
        """
        datasets = []

        number_datasets = click.prompt(
            "Number of datasets (more than one means multitask learning)",
            type=click.IntRange(min=1),
            default=1,
        )
        for i in range(number_datasets):
            data_txt = click.prompt(f"Path to the data set (TXT) no {i + 1}", type=str)
            data_dir = click.prompt(
                f"Where to save the processed data set no {i + 1}", type=str
            )

            # weight does not need to be queried if there's only one dataset
            if number_datasets == 1:
                weight = 1
            else:
                weight = click.prompt(
                    f"Training weight for data set no {i + 1}",
                    type=click.IntRange(min=1),
                )
            datasets.append(
                DatasetOptions(
                    txt_path=data_txt,
                    processed_path=data_dir,
                    weight=weight,
                    augment=self._maybe_get_augment_options(i + 1),
                )
            )

        return datasets

    def _maybe_get_context_options(self) -> Optional[ContextOptions]:
        if self.model_task != "context":
            return None

        tagging_batch_size = click.prompt(
            "Batch size for generating context prediction data",
            type=int,
            default=_CONTEXT_DATA_BATCH_SIZE,
        )
        return ContextOptions(tagging_batch_size=tagging_batch_size)

    def _maybe_get_augment_options(self, dataset_no: int) -> Optional[AugmentOptions]:
        augment = click.confirm(
            f"Would you like to augment the data set {dataset_no}?", default=False
        )
        if not augment:
            return None
        n_augmentations = click.prompt(
            "Number of augmentations per sample", type=click.IntRange(min=1)
        )
        return AugmentOptions(number_augmentations=n_augmentations)

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
    def _prepare_data_cmd(dataset: DatasetOptions, prepare_seed: int) -> str:
        command = (
            f"rxn-prepare-data --input_data {dataset.txt_path} "
            f"--output_dir {dataset.processed_path} "
        )
        if prepare_seed != defaults.SEED:
            command += f"--split_seed {prepare_seed} "
        return command

    def _augment_cmd(self, dataset: DatasetOptions) -> Optional[str]:
        if dataset.augment is None:
            return None
        return (
            f"rxn-onmt-augment --data_dir {dataset.processed_path} --model_task "
            f"{self.model_task} -n {dataset.augment.number_augmentations}"
        )

    def _prepare_context_data_cmd(self, data_dir: str) -> str:
        if self.context_options is None:
            raise RuntimeError("Context options not defined.")

        command = f"rxn-create-context-dataset --data_dir {data_dir} "
        if self.context_options.tagging_batch_size != _CONTEXT_DATA_BATCH_SIZE:
            command += f"--batch_size {self.context_options.tagging_batch_size} "
        return command

    def _data_weights(self) -> str:
        data_weights = ""
        if len(self.datasets) > 1:
            for dataset in self.datasets:
                data_weights += f"--data_weights {dataset.weight} "
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
    if tp.model_task == "context":
        print(
            "# 1b) Prepare context prediction data (requires rxn-context-prediction package)"
        )
        for prepare_context_cmd in tp.prepare_context_data_cmd():
            print(prepare_context_cmd)
        print()
    if any(dataset.augment is not None for dataset in tp.datasets):
        print("# 1c) Augment the data")
        for augment_cmd in tp.augment_data_cmd():
            print(augment_cmd)
        print()
    print(f"# 2) Preprocess the data with OpenNMT\n{tp.preprocess_cmd()}\n")
    print(f"# 3) Train the model\n{tp.train_or_finetune_cmd()}\n")
    print(f"# 4) If necessary: continue training\n{tp.continue_training_cmd()}")


if __name__ == "__main__":
    main()
