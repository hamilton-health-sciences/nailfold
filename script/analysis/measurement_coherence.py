"""Assesses the coherence between hidden model states and measurements."""

import argparse

import os

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from scipy.stats import chi2

import statsmodels.api as sm

from nailfold import config
from nailfold.data import RepresentationsDataModule


def glm_lrt(model0, model):
    loglik_0 = model0.llf
    loglik = model.llf
    stat = -2 * (loglik_0 - loglik)
    df_diff = model.df_model - model0.df_model
    pval = 1 - chi2.cdf(stat, df=df_diff)

    return stat, pval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--measure", type=str, required=True,
                        choices=["capillary_count", "lengths", "widths"])
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--input_predictions", type=str, default=None)
    parser.add_argument("--input_representations", type=str, default=None)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    if args.input_predictions is not None:
        predictions_fn = args.input_predictions
    else:
        predictions_fn = os.path.join(config.RESULTS_ROOT,
                                      args.outcome,
                                      "predictions.csv")
    predictions = pd.read_csv(predictions_fn).iloc[:, 1:]
    img_relpath = predictions["image_filepath"].apply(
        lambda s: "/".join(s.split("/")[-3:])
    )
    predictions["image_path"] = img_relpath
    predictions = predictions.drop("image_filepath", axis=1)
    predictions = predictions.set_index("image_path")

    if args.measure in ["lengths", "widths"]:
        reduce_fn = np.median
    else:
        reduce_fn = lambda x: x

    if args.input_representations is not None:
        representations_fn = args.input_representations
    else:
        representations_fn = os.path.join(
            config.RESULTS_ROOT,
            args.outcome,
            f"representations_repetition={args.repetition}_fold={args.fold}.pkl"
        )

    dm = RepresentationsDataModule(
        data_root=config.DATA_ROOT,
        representations_fn=representations_fn,
        measure=args.measure,
        reduce_fn=reduce_fn,
        batch_size=config.MAX_REPRESENTATION_BATCH_SIZE,
        normalize=False
    )
    dm.setup()

    train_data = next(iter(dm.train_dataloader()))
    val_data = next(iter(dm.val_dataloader()))
    test_data = next(iter(dm.test_dataloader()))
    measurements = pd.DataFrame({
        k: np.concatenate(
            [subset[k] for subset in [train_data, val_data, test_data]]
        )
        for k in ["image_path", "measure", "age", "gender"]
    }).set_index("image_path")

    representations = np.concatenate([
        subset["representation"]
        for subset in [train_data, val_data, test_data]
    ], axis=0)

    joined = measurements.join(predictions)

    # Subset to fully observed
    sel = ~pd.isnull(joined["y"])
    joined = joined.loc[sel, :]
    representations = representations[sel, :]

    # Compute the PCA such that 95% of the variance in the representations is
    # explained
    pca = PCA(n_components=0.95, svd_solver="full").fit(representations)
    scores = pca.transform(representations)

    # Get variables
    baseline = np.asarray(joined[["age", "gender"]])
    measure = np.asarray(joined["measure"])
    outcome = np.asarray(joined["y"])
    baseline_outcome = np.concatenate((baseline, outcome[:, np.newaxis]),
                                      axis=1)

    print(f"{args.outcome}: repetition {args.repetition}, fold {args.fold} "
          f" ({args.measure})")

    # Fit a LR model to: binarized measure ~ baseline
    baseline_model = sm.Logit(
        measure < args.threshold,
        sm.add_constant(baseline)
    ).fit(disp=0)

    # Fit a LR model to: binarized measure ~ baseline + PC scores
    baseline_scores_model = sm.Logit(
        measure < args.threshold,
        sm.add_constant(np.concatenate((baseline, scores), axis=1))
    ).fit(disp=0)

    _, pval = glm_lrt(baseline_model, baseline_scores_model)
    print("\tLRT p-value (scores + baseline vs. baseline): ", pval)

    # Fit a LR model to: binarized measure ~ baseline + outcome
    baseline_outcome_model = sm.Logit(
        measure < args.threshold,
        sm.add_constant(baseline_outcome)
    ).fit(disp=0)

    # Fit a GLM to: binarized measure ~ baseline + outcome + PC scores
    baseline_outcome_scores_model = sm.Logit(
        measure < args.threshold,
        sm.add_constant(
            np.concatenate((baseline_outcome, scores), axis=1)
        )
    ).fit(disp=0)

    # Conduct an LRT to determine if the representation PCs contain additional
    # information about the measure compared to just the outcome
    _, pval = glm_lrt(baseline_outcome_model, baseline_outcome_scores_model)
    print("\tNumber of PCs selected: ", scores.shape[1])
    print(
        "\tLRT p-value (scores + baseline + outcome vs. baseline + outcome): ",
        pval
    )


if __name__ == "__main__":
    main()
