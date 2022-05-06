# Training scripts

These training scripts were copied/adapted from the `zhc2` cluster.

The training scripts in this directory are to be executed in a Python environment where our fork of `OpenNMT-py`, https://github.ibm.com/rxn/OpenNMT-py, has been installed.

The scripts are compatible with the official `OpenNMT-py`, as long as it is an older version (< 2.0), and the options related to `wandb` are not used.

For use of `wandb` for following the progress, the two following environment variables must be defined: `WANDB_BASE_URL`, `WANDB_API_KEY`.
