from typing import Any, List, Tuple

from rxn.utilities.files import PathLike

from rxn_onmt_utils.rxn_models.utils import RxnCommand, preprocessed_id_names


class Arg:
    """
    Represents an argument to be given for the onmt_train command.

    Attributes:
        key: argument name (i.e. what is forwarded to onmt_train, without the dash).
        default: default value that we use for that argument in the RXN universe.
            None indicates that this argument must be provided explicitly, an
            empty string is used for boolean args not requiring a value.
        needed_for: what commands this argument is needed for (train, finetune, etc.)
    """

    def __init__(self, key: str, default: Any, needed_for: RxnCommand):
        self.key = key
        self.default = default
        self.needed_for = needed_for


ONMT_TRAIN_ARGS: List[Arg] = [
    Arg("accum_count", "4", RxnCommand.TCF),
    Arg("adam_beta1", "0.9", RxnCommand.TF),
    Arg("adam_beta2", "0.998", RxnCommand.TF),
    Arg("batch_size", None, RxnCommand.TCF),
    Arg("batch_type", "tokens", RxnCommand.TCF),
    Arg("data", None, RxnCommand.TCF),
    Arg("decay_method", "noam", RxnCommand.TF),
    Arg("decoder_type", "transformer", RxnCommand.T),
    Arg("dropout", None, RxnCommand.TCF),
    Arg("encoder_type", "transformer", RxnCommand.T),
    Arg("global_attention", "general", RxnCommand.T),
    Arg("global_attention_function", "softmax", RxnCommand.T),
    Arg("heads", None, RxnCommand.T),
    Arg("keep_checkpoint", "20", RxnCommand.TCF),
    Arg("label_smoothing", "0.0", RxnCommand.TCF),
    Arg("layers", None, RxnCommand.T),
    Arg("learning_rate", None, RxnCommand.TF),
    Arg("max_generator_batches", "32", RxnCommand.TCF),
    Arg("max_grad_norm", "0", RxnCommand.TF),
    Arg("normalization", "tokens", RxnCommand.TCF),
    Arg("optim", "adam", RxnCommand.TF),
    Arg("param_init", "0", RxnCommand.T),
    Arg("param_init_glorot", "", RxnCommand.T),  # note: empty means "nothing"
    Arg("position_encoding", "", RxnCommand.T),  # note: empty means "nothing"
    Arg("report_every", "1000", RxnCommand.TCF),
    Arg("reset_optim", None, RxnCommand.CF),
    Arg("rnn_size", None, RxnCommand.TF),
    Arg("save_checkpoint_steps", "5000", RxnCommand.TCF),
    Arg("save_model", None, RxnCommand.TCF),
    Arg("seed", None, RxnCommand.TCF),
    Arg("self_attn_type", "scaled-dot", RxnCommand.T),
    Arg("share_embeddings", "", RxnCommand.T),  # note: empty means "nothing"
    Arg("train_from", None, RxnCommand.CF),
    Arg("train_steps", None, RxnCommand.TCF),
    Arg("transformer_ff", None, RxnCommand.T),
    Arg("valid_batch_size", "8", RxnCommand.TCF),
    Arg("warmup_steps", None, RxnCommand.TF),
    Arg("word_vec_size", None, RxnCommand.T),
]


class OnmtTrainCommand:
    """
    Class to build the onmt_command for training models, continuing the
    training, or finetuning.
    """

    def __init__(
        self,
        command_type: RxnCommand,
        no_gpu: bool,
        data_weights: Tuple[int, ...],
        **kwargs: Any,
    ):
        self._command_type = command_type
        self._no_gpu = no_gpu
        self._data_weights = data_weights
        self._kwargs = kwargs

    def _build_cmd(self) -> List[str]:
        """
        Build the base command.
        """
        command = ["onmt_train"]
        for arg in ONMT_TRAIN_ARGS:
            arg_given = arg.key in self._kwargs

            if self._command_type not in arg.needed_for:
                # Check that the arg was not given; then go to the next argument.
                if arg_given:
                    raise ValueError(
                        f'"{arg.key}" value given, but not necessary for {command}'
                    )
                continue

            needs_value = arg.default is None

            # Case 1: needs value but nothing given
            if needs_value and not arg_given:
                raise ValueError(f"No value given for {arg.key}")
            # Case 2: needs value and something given
            elif needs_value and arg_given:
                value = str(self._kwargs[arg.key])
            # Case 3: does not need value, something given
            elif arg_given:
                raise ValueError(f"Value given for {arg.key}, but nothing was needed.")
            # Case 4: does not need value and nothing given
            else:
                value = str(arg.default)

            # Add the args to the command. Note: if the value is the empty string,
            # do not add anything (typically for boolean args)
            command.append(f"-{arg.key}")
            if value != "":
                command.append(value)

        command += self._args_for_gpu()
        command += self._args_for_data_weights()

        return command

    def _args_for_gpu(self) -> List[str]:
        if self._no_gpu:
            return []
        return ["-gpu_ranks", "0"]

    def _args_for_data_weights(self) -> List[str]:
        if not self._data_weights:
            return []

        n_additional_datasets = len(self._data_weights) - 1
        data_ids = preprocessed_id_names(n_additional_datasets)
        return [
            "-data_ids",
            *data_ids,
            "-data_weights",
            *(str(weight) for weight in self._data_weights),
        ]

    def cmd(self) -> List[str]:
        """
        Return the "raw" command for executing onmt_train.
        """
        return self._build_cmd()

    def save_to_config_cmd(self, config_file: PathLike) -> List[str]:
        """
        Return the command for saving the config to a file.
        """
        return self._build_cmd() + ["-save_config", str(config_file)]
        pass

    @staticmethod
    def execute_from_config_cmd(config_file: PathLike) -> List[str]:
        """
        Return the command for executing onmt_train with values read from the config.
        """
        return ["onmt_train", "-config", str(config_file)]

    @classmethod
    def train(
        cls,
        batch_size: int,
        data: PathLike,
        dropout: float,
        heads: int,
        layers: int,
        learning_rate: float,
        rnn_size: int,
        save_model: PathLike,
        seed: int,
        train_steps: int,
        transformer_ff: int,
        warmup_steps: int,
        word_vec_size: int,
        no_gpu: bool,
        data_weights: Tuple[int, ...],
    ) -> "OnmtTrainCommand":
        return cls(
            command_type=RxnCommand.T,
            no_gpu=no_gpu,
            data_weights=data_weights,
            batch_size=batch_size,
            data=data,
            dropout=dropout,
            heads=heads,
            layers=layers,
            learning_rate=learning_rate,
            rnn_size=rnn_size,
            save_model=save_model,
            seed=seed,
            train_steps=train_steps,
            transformer_ff=transformer_ff,
            warmup_steps=warmup_steps,
            word_vec_size=word_vec_size,
        )

    @classmethod
    def continue_training(
        cls,
        batch_size: int,
        data: PathLike,
        dropout: float,
        save_model: PathLike,
        seed: int,
        train_from: PathLike,
        train_steps: int,
        no_gpu: bool,
        data_weights: Tuple[int, ...],
    ) -> "OnmtTrainCommand":
        return cls(
            command_type=RxnCommand.C,
            no_gpu=no_gpu,
            data_weights=data_weights,
            batch_size=batch_size,
            data=data,
            dropout=dropout,
            reset_optim="none",
            save_model=save_model,
            seed=seed,
            train_from=train_from,
            train_steps=train_steps,
        )

    @classmethod
    def finetune(
        cls,
        batch_size: int,
        data: PathLike,
        dropout: float,
        learning_rate: float,
        rnn_size: int,
        save_model: PathLike,
        seed: int,
        train_from: PathLike,
        train_steps: int,
        warmup_steps: int,
        no_gpu: bool,
        data_weights: Tuple[int, ...],
    ) -> "OnmtTrainCommand":
        return cls(
            command_type=RxnCommand.F,
            no_gpu=no_gpu,
            data_weights=data_weights,
            batch_size=batch_size,
            data=data,
            dropout=dropout,
            learning_rate=learning_rate,
            reset_optim="all",
            rnn_size=rnn_size,
            save_model=save_model,
            seed=seed,
            train_from=train_from,
            train_steps=train_steps,
            warmup_steps=warmup_steps,
        )
