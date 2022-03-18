"""Use Ray Tune to perform hyperparameter optimization."""

from typing import Optional

from warnings import warn

from functools import partial

import argparse

from uuid import uuid4

import torch

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from nailfold import config as global_config
from nailfold.data import NailfoldDataModule
from nailfold.models import TorchvisionWrapper
from nailfold.trainer import Trainer


def train_run(config: dict,
              outcome: str,
              fold: int,
              repetition: int,
              condition_on: str,
              init_seed: int,
              warmup_epochs: int) -> None:
    """
    Fit the model using the given parameters.

    Parameters:
        config: The chosen hyperparameters for this fit.
        outcome: The outcome of interest.
        fold: The held-out fold index for k-fold CV.
        repetition: The CV repetition index, also used to seed data splitting.
        condition_on: The covariate to condition on. If None, this is ignored.
        init_seed: The seed used for the model initialization step.
        warmup_epochs: The number of epochs to warm up the learning rate when
                       using linear learning rate warmup in conjunction with
                       cosine annealing.
    """
    # DataModule
    dm = NailfoldDataModule(binary_covariate_name=outcome,
                            split_seed=repetition,
                            condition_on=condition_on,
                            fold=fold,
                            batch_size=config["bs"])
    dm.setup()

    # Model
    seed_everything(init_seed)
    torch.set_deterministic(True)
    model = TorchvisionWrapper(
        architecture=config["arch"],
        pretrained=config["pretrained"],
        conditional_cov=(condition_on is not None),
        optimizer=config["optimizer"],
        learning_rate=config["lr"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
        weight_decay=config["wd"],
        exclude_biases_from_weight_decay=config["wd_exclude_biases"],
        cosine_anneal_lr=config["cosine_anneal"],
        head_dropout_prob=config["head_dropout_prob"],
        head_hidden_dim=config["head_hidden_dim"],
        case_weight=dm.train_set.case_weight,
        mode=dm.mode,
        eval_outcome_transform=None,
        split_seed=repetition,
        init_seed=init_seed,
        warmup_steps=(warmup_epochs * len(dm.train_dataloader())),
        max_steps=(global_config.NUM_EPOCHS * len(dm.train_dataloader()))
    )

    # Callbacks
    callbacks = []

    callbacks.append(
        # PyTorch Lightning -> Tune reporting
        TuneReportCallback(
            {
                "val_loss": "validation_loss",
                "val_auroc": "validation_patient_auroc",
                "val_img_auroc": "validation_img_auroc"
            },
            on="validation_end"
        )
    )

    if config["cosine_anneal"]:
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    ckpt_fn = ("{epoch}-{validation_patient_auroc:.5f}"
               "-{validation_img_auroc:.5f}-{validation_loss:.5f}")
    callbacks.append(
        # Validation loss-based checkpointing (not currently used downstream)
        ModelCheckpoint(
            monitor="validation_loss",
            save_top_k=5,
            period=1,
            mode="min",
            filename=ckpt_fn
        )
    )
    callbacks.append(
        # Validation image AUROC-based checkpointing (used for model selection)
        ModelCheckpoint(
            monitor="validation_img_auroc",
            save_top_k=5,
            period=1,
            save_last=True,
            mode="max",
            filename=ckpt_fn
        )
    )

    # Trainer
    trainer = Trainer(
        max_epochs=global_config.NUM_EPOCHS,
        gpus=1,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(),
            name="",
            version=""
        ),
        progress_bar_refresh_rate=0,
        callbacks=callbacks
    )

    # Train/validate
    trainer.fit(model, dm)


def trial_namer(trial) -> str:
    """
    Generate a name for a Ray Tune Trial based on its hyperparameters.
    """
    arch = trial.config["arch"]
    bs = trial.config["bs"]
    lr = trial.config["lr"]
    wd = trial.config["wd"]
    pfx = str(uuid4())[:8]

    name = "arch=%s_bs=%d_lr=%.8f_wd=%.8f_%s" % (arch, bs, lr, wd, pfx)

    return name


def tune_run(output_dir: str,
             num_samples: int,
             warmup_epochs: int,
             outcome: str,
             fold: int,
             repetition: int,
             condition_on: Optional[str],
             tune_seed: int,
             init_seed: int,
             resume_errored: bool) -> None:
    """
    Run hyperparameter optimization.

    Parameters:
        output_dir: The output directory to store run information.
        num_samples: The number of models to fit.
        warmup_epochs: The number of warmup epochs to use when using cosine
                       annealing with linear warmup.
        outcome: The outcome of interest.
        fold: The index of the held-out fold for k-fold CV.
        repetition: The CV index, also used to seed splitting.
        condition_on: If given, the covariate to condition on.
        tune_seed: The random seed for the hyperparameter tuner.
        init_seed: The random seed used for model initialization.
        resume_errored: If True, will only run trials that previously errored.
    """
    # The search space
    tune_config = {
        "arch": tune.choice(
            ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"]
        ),
        "bs": tune.choice([2, 4, 8]),
        "lr": tune.loguniform(1e-6, 1e-3),
        "wd": tune.loguniform(1e-6, 1e-3),
        # These parameters are static i.e. not optimized
        "pretrained": True,
        "wd_exclude_biases": True,
        "optimizer": "adam",
        "momentum": 0.,
        "nesterov": False,
        "cosine_anneal": True,
        # These parameters are ignored for non-conditional models
        "head_dropout_prob": 0.,
        "head_hidden_dim": 0
    }
    if condition_on is not None:
        tune_config["head_dropout_prob"] = tune.choice([0., 0.2, 0.5])
        tune_config["head_hidden_dim"] = tune.choice([8, 16, 64, 128])

    # HyperOpt implementation in Ray uses tree-structured Parzen estimators
    # to suggest the next hyperparameter setting
    searcher = HyperOptSearch(
        metric="val_img_auroc",
        mode="max",
        random_state_seed=(tune_seed % (2**32 - 1))
    )

    # Default scheduler in Ray, which "aggressively" terminates under-
    # performing models
    scheduler = AsyncHyperBandScheduler(
        metric="val_img_auroc",
        mode="max",
        max_t=global_config.NUM_EPOCHS,
        grace_period=global_config.MIN_TRAINING_EPOCHS
    )

    # Report quantities to the terminal
    parameter_columns = ["arch", "bs", "lr", "wd"]
    if condition_on is not None:
        parameter_columns += ["head_dropout_prob", "head_hidden_dim"]
    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=["train_img_auroc", "val_patient_auroc",
                        "val_img_auroc"]
    )

    if resume_errored:
        resume = "ERRORED_ONLY"
    else:
        resume = False

    if condition_on is not None:
        name = "%s_cond_%s_repetition=%d_fold=%d" % (outcome,
                                                     condition_on,
                                                     repetition,
                                                     fold)
    else:
        name = "%s_repetition=%d_fold=%d" % (outcome,
                                             repetition,
                                             fold)


    # Run tuning
    tune.run(
        partial(
            train_run,
            outcome=outcome,
            fold=fold,
            repetition=repetition,
            condition_on=condition_on,
            init_seed=init_seed,
            warmup_epochs=warmup_epochs,
        ),
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        config=tune_config,
        num_samples=num_samples,
        search_alg=searcher,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=output_dir,
        name=name,
        trial_name_creator=trial_namer,
        resume=resume
    )


def main():
    """
    The main entry point for the CLI interface for tuning models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--condition_on", type=str, default=None)

    parser.add_argument("--resume_errored", action="store_true",
                        default=False)

    parser.add_argument("--init_seed", type=int, default=1)

    parser.add_argument("--warmup_epochs", type=int, default=0)

    parser.add_argument("--num_samples", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="./ray_logs")

    args = parser.parse_args()

    if args.num_samples is None:
        if args.condition_on is None:
            num_samples = global_config.NUM_TUNING_FITS_NONCONDITIONAL
        else:
            num_samples = global_config.NUM_TUNING_FITS_CONDITIONAL
    else:
        num_samples = args.num_samples

    # Launch Ray
    warn("By default, the dashboard will be accessible from any host on the "
         "network. This is probably a bad idea in uncontrolled environments.")
    ray.init(dashboard_host="0.0.0.0")

    # Generate an outcome-, repetition-, and fold-specific tuning seed so that
    # the hyperparameter tuning search path is independent across outcomes,
    # repetitions, and folds (i.e. no potential information leakage between
    # them)
    outcome_seed = int.from_bytes(args.outcome.encode(), "little")
    tune_seed = args.repetition * global_config.NUM_FOLDS + args.fold
    outcome_tune_seed = outcome_seed + tune_seed

    tune_run(
        num_samples=num_samples,
        warmup_epochs=args.warmup_epochs,
        outcome=args.outcome,
        repetition=args.repetition,
        fold=args.fold,
        condition_on=args.condition_on,
        tune_seed=outcome_tune_seed,
        init_seed=args.init_seed,
        output_dir=args.output_dir,
        resume_errored=args.resume_errored
    )


if __name__ == "__main__":
    main()
