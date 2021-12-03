import argparse
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster

from datasets.kinetics import KineticsDataModule
from datasets.kinetics_dummy import DummyKineticsDataModule
from models.brave import Brave


def train(args, cluster):
    print(args)


if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy="random_search")
    parser.add_argument("--test_tube_exp_name", default="cgan_celeba")
    parser.add_argument("--log_path", default="./runs_celeba_slurm")
    parser.add_argument("--dataroot", default="data", help="path to dataset")
    parser.add_argument(
        "--dataset-size", default=50000, type=int, help="Number of real images to use"
    )
    parser.add_argument(
        "--noise-level",
        default=0.0,
        type=float,
        help="Percentage of labels to randomize",
    )
    parser.add_argument(
        "--num-workers", default=1, type=int, help="number of data loading workers"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=30, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0008, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="check a single training cycle works"
    )
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--logdir", default=None, help="folder to output images and model checkpoints"
    )
    parser.add_argument(
        "--save-frequency", default=50, type=int, help="number of batches between saves"
    )
    parser.add_argument("--seed", type=int, default=1, help="manual seed")
    args = parser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=args,
        log_path=args.log_path,
        python_cmd="python",
    )

    # Add commands to the non-SLURM portion.
    cluster.add_command("cd /vision/u/naagnes/github/few-shot-cgan")
    cluster.add_command("source .svl/bin/activate")

    # SLURM commands.
    cluster.add_slurm_cmd(cmd="partition", value="svl", comment="")
    cluster.add_slurm_cmd(cmd="qos", value="normal", comment="")
    cluster.add_slurm_cmd(cmd="time", value="48:00:00", comment="")
    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=1, comment="")
    cluster.add_slurm_cmd(cmd="cpus-per-task", value=32, comment="")
    cluster.add_slurm_cmd(cmd="mem", value="120G", comment="")

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.gpu_type = "titanrtx"

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(train, nb_trials=1, job_name="cgan_celeba")
