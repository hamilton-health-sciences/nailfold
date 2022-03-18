"""Extract the predictions from the best models found by Ray Tune."""

from typing import Dict, Any

import argparse

from functools import reduce

import os

import numpy as np

import pandas as pd

from nailfold import config
from nailfold.data import NailfoldDataModule
from nailfold.trainer import Trainer
from nailfold.utils import get_best_model


def get_experiment_predictions(logs_root: str,
                               outcome: str,
                               condition_on: str,
                               split_seed: int,
                               fold: int) -> Dict[str, Any]:
    """
    Get the predictions for a specific experiment (outcome, split, and fold).

    Args:
        logs_root: The root path of all Ray Tune logs.
        outcome: Binary outcome of interest.
        split_seed: The seed used to split the data (repetition number).
        fold: The fold of the data held out for final evaluation.

    Returns:
        predictions: A dictionary with keys for `id` (patient ID), `y` (ground-
                     truth labels), and `yhat` (predicted logits). Each value
                     should have length equal to the number of test images.
    """
    if condition_on is not None:
        outcome_pfx = "%s_cond_%s" % (outcome, condition_on)
    else:
        outcome_pfx = outcome
    model = get_best_model(logs_root, outcome_pfx, split_seed, fold)
    dm = NailfoldDataModule(binary_covariate_name=outcome,
                            condition_on=condition_on,
                            split_seed=split_seed,
                            fold=fold)

    trainer = Trainer(gpus=1)
    results = trainer.test(model, datamodule=dm, verbose=False)[0]

    predictions = {
        "id": results["id"],
        "image_filepath": results["image_filepath"],
        "y": results["y"],
        "yhat": results["yhat"],
    }

    return predictions


def get_outcome_predictions(logs_root, outcome, condition_on, split_seeds):
    """Get all predictions from all splits for a given outcome.

    Parameters:
        logs_root : str
            The root path of all Ray Tune logs.
        outcome : str
            The binary outcome of interest.
        split_seeds : Sequence[int]
            A list of split seeds (repetition numbers) to use to compute the
            results.

    Returns:
        predictions : DataFrame
            A data frame consisting of the predictions for each split and each
            fold.
    """
    all_predictions = []
    for split_seed in split_seeds:
        for fold in range(config.NUM_FOLDS):
            try:
                predictions = get_experiment_predictions(logs_root,
                                                         outcome,
                                                         condition_on,
                                                         split_seed,
                                                         fold)
                predictions["fold"] = [fold] * len(predictions["id"])
                predictions["cvrep"] = [split_seed] * len(predictions["id"])
                all_predictions.append(predictions)
            except Exception:
                print(
                    "Error getting results for repetition %d, fold %d" %
                    (split_seed, fold)
                )
                raise

    predictions = reduce(
        lambda x, y: {k: np.concatenate((x[k], y[k])) for k in x},
        all_predictions
    )
    predictions = pd.DataFrame(predictions)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--condition_on", type=str, default=None)
    parser.add_argument("--split_seeds", type=int, nargs="+",
                        default=list(range(1)))
    args = parser.parse_args()

    if args.condition_on is not None:
        outcome_desc = "%s_cond_%s" % (args.outcome, args.condition_on)
    else:
        outcome_desc = args.outcome

    # Ensure there"s a directory to put the results in
    results_root = os.path.join(config.RESULTS_ROOT, outcome_desc)
    os.makedirs(results_root, exist_ok=True)

    predictions = get_outcome_predictions(config.RAY_LOGS_ROOT,
                                          args.outcome,
                                          args.condition_on,
                                          args.split_seeds)
    output_filename = os.path.join(results_root, "predictions.csv")
    predictions.to_csv(output_filename)


if __name__ == "__main__":
    main()
