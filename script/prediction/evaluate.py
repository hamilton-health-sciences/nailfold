"""Analyze predictions to get performance tables/plots for each outcome."""

from typing import Tuple

from functools import partial

import argparse

import os

import traceback

import torch

import numpy as np

import pandas as pd

from plotnine import (ggplot, geom_line, geom_ribbon, ylim, geom_segment,
                      ggtitle, aes, theme_bw)

from sklearn.metrics import roc_auc_score, average_precision_score

from nailfold import config
from nailfold.evaluation_utils import (sensitivity,
                                       specificity,
                                       roc_curve,
                                       pr_curve,
                                       baseline_x as curve_values_x,
                                       compute_bootstraps,
                                       positive_class_frequency)


def compute_statistics(sub_predictions: pd.DataFrame):
    """
    Compute the statistics for a given batch of predictions.

    Args:
        sub_predictions: A table of logits (`yhat`) and ground-truth binary
                         labels (`y`).

    Returns:
        results: Summary performance statistics for the samples in the given
                 predictions DataFrame.
    """
    sub_predictions["yprob"] = torch.sigmoid(
        torch.from_numpy(np.asarray(sub_predictions["yhat"]))
    )
    patient_predictions = sub_predictions.groupby("id").agg("mean")

    if len(np.unique(sub_predictions["y"])) == 1:
        print("Only one unique outcome in the whole fold. This is likely due "
              "to the subanalysis option. Consider not running this "
              "subanalysis.")
        return None

    metrics = {"AUROC": roc_auc_score, "AUPR": average_precision_score,
               "sensitivity": sensitivity, "specificity": specificity,
               "roc_curve": roc_curve, "pr_curve": pr_curve,
               "positive_class_frequency": positive_class_frequency}
    results = {}
    for metric, metric_fn in metrics.items():
        results["%s.img" % metric] = metric_fn(sub_predictions["y"],
                                               sub_predictions["yhat"])
        bootstrapped_metric_img = compute_bootstraps(
            np.asarray(sub_predictions["yhat"]).astype(float),
            np.asarray(sub_predictions["y"]).astype(int),
            stat_func=metric_fn
        )
        results["%s.img.bootstraps" % metric] = bootstrapped_metric_img
        ci_lower, ci_upper = np.quantile(bootstrapped_metric_img,
                                         [0.025, 0.975],
                                         axis=0)
        results["%s.img.CI.lower" % metric] = ci_lower
        results["%s.img.CI.upper" % metric] = ci_upper

        results["%s.patient" % metric] = metric_fn(
            patient_predictions["y"], patient_predictions["yprob"]
        )
        bootstrapped_metric_patient = compute_bootstraps(
            np.asarray(patient_predictions["yprob"]).astype(float),
            np.asarray(patient_predictions["y"]).astype(int),
            stat_func=metric_fn
        )
        results["%s.patient.bootstraps" % metric] = bootstrapped_metric_patient
        ci_lower, ci_upper = np.quantile(bootstrapped_metric_patient,
                                         [0.025, 0.975],
                                         axis=0)
        results["%s.patient.CI.lower" % metric] = ci_lower
        results["%s.patient.CI.upper" % metric] = ci_upper

    return pd.Series(results)


def get_curve_plots(stats: pd.DataFrame,
                    rep_stats: pd.DataFrame):
    if len(rep_stats) > 1:
        raise ValueError("Not sure how to compute CIs for ROC curves for more "
                         "than 1 CV repetition.")
    plots = {}
    mean_plot_metrics = {}

    # ROC curve
    mean_roc_curve = np.array(stats["roc_curve.patient"].tolist()).mean(axis=0)
    roc_curve_lower = rep_stats["roc_curve.patient.CI.lower"][0]["wrapped"]
    roc_curve_upper = rep_stats["roc_curve.patient.CI.upper"][0]["wrapped"]

    # Identify intercepts for Youden's index on patient curve
    youden_idx = np.argmax(mean_roc_curve - curve_values_x)
    youden_roc_x = curve_values_x[youden_idx]
    youden_sensitivity = mean_roc_curve[youden_idx]
    youden_recall_idx = np.argmin(np.abs(curve_values_x - youden_sensitivity))
    youden_pr_x = curve_values_x[youden_recall_idx]

    mean_plot_metrics["sensitivity"] = youden_sensitivity
    mean_plot_metrics["specificity"] = 1. - youden_roc_x

    # Identify intercepts for Youden's index on image curve
    mean_roc_curve = np.array(stats["roc_curve.img"].tolist()).mean(axis=0)
    youden_idx = np.argmax(mean_roc_curve - curve_values_x)
    youden_roc_x = curve_values_x[youden_idx]
    youden_sensitivity = mean_roc_curve[youden_idx]
    youden_recall_idx = np.argmin(np.abs(curve_values_x - youden_sensitivity))
    youden_pr_x = curve_values_x[youden_recall_idx]

    mean_plot_metrics["sensitivity.img"] = youden_sensitivity
    mean_plot_metrics["specificity.img"] = 1. - youden_roc_x


    plot_df = pd.DataFrame({
        "Sensitivity": mean_roc_curve,
        "1 - Specificity": curve_values_x,
        "lower": roc_curve_lower,
        "upper": roc_curve_upper
    })
    plots["ROC_curve"] = (
        ggplot(plot_df) +
            geom_line(aes(x="1 - Specificity", y="Sensitivity")) +
            geom_ribbon(aes(x="1 - Specificity", ymin="lower", ymax="upper"),
                        alpha=0.25) +
            ylim([0, 1]) +
            geom_segment(x=0, y=0, xend=1, yend=1, linetype="dashed",
                         size=0.25) +
            geom_segment(x=youden_roc_x, y=0, xend=youden_roc_x,
                         yend=youden_sensitivity, linetype="dotted") +
            theme_bw()
    )

    # PR curve
    mean_pr_curve = np.array(stats["pr_curve.patient"].tolist()).mean(axis=0)
    pr_curve_lower = rep_stats["pr_curve.patient.CI.lower"][0]["wrapped"]
    pr_curve_upper = rep_stats["pr_curve.patient.CI.upper"][0]["wrapped"]
    youden_pr_y = mean_pr_curve[youden_recall_idx]
    plot_df = pd.DataFrame({
        "Precision": mean_pr_curve,
        "Recall": curve_values_x,
        "lower": pr_curve_lower,
        "upper": pr_curve_upper
    })
    plots["PR_curve"] = (
        ggplot(plot_df) +
            geom_line(aes(x="Recall", y="Precision")) +
            geom_ribbon(aes(x="Recall", ymin="lower", ymax="upper"),
                        alpha=0.25) +
            ylim([0, 1]) +
            geom_segment(x=youden_pr_x, y=0, xend=youden_pr_x,
                         yend=youden_pr_y, linetype="dotted") +
            geom_segment(x=0,
                         y=rep_stats["positive_class_frequency.patient"],
                         xend=1,
                         yend=rep_stats["positive_class_frequency.patient"],
                         linetype="dashed") +
            theme_bw()
    )

    return plots, mean_plot_metrics


def extract_metrics(predictions) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the metrics of interest for a given outcome.

    Args:
        predictions: DataFrame of predictions, with columns `cvrep` (CV
                     repetition), `fold` (fold in CV repetition), `yhat`
                     (prediction logits), `y` (ground-truth labels), and `id`
                     (patient IDs).

    Returns:
        stats: Fold-level performance statistics.
        repetition_stats: CV repetition-level performance statistics.
    """
    def _compute_percentile_bootstraps(column, q):
        x = np.array(column.tolist())
        xmean = x.mean(axis=0)
        xquant = np.quantile(xmean, q, axis=0)
        if np.prod(xquant.shape) > 1:
            xquant = {"wrapped": xquant}

        return xquant
    stats = predictions.groupby(["cvrep", "fold"]).apply(compute_statistics)
    rep_stats = stats[[c for c in stats.columns if "bootstrap" in c]]
    rep_stats = rep_stats.groupby("cvrep")
    ci_lowers = rep_stats.agg(partial(_compute_percentile_bootstraps, q=0.025))
    ci_lowers.columns = [c.replace("bootstraps", "CI.lower")
                         for c in ci_lowers.columns]
    ci_uppers = rep_stats.agg(partial(_compute_percentile_bootstraps, q=0.975))
    ci_uppers.columns = [c.replace("bootstraps", "CI.upper")
                         for c in ci_uppers.columns]

    cols = stats.columns
    repetition_stats = stats[
        [c for c in cols if "CI" not in c and "bootstraps" not in c]
    ].groupby("cvrep").agg("mean")
    repetition_stats = pd.concat([repetition_stats, ci_lowers, ci_uppers],
                                 axis=1)
    repetition_stats = repetition_stats.reindex(
        sorted(repetition_stats.columns),
        axis=1
    )

    stats = stats[[c for c in stats.columns if "bootstraps" not in c]]

    plots, mean_curve_metrics = get_curve_plots(stats, repetition_stats)

    stats = stats.drop(
        [c for c in stats.columns if "curve" in c],
        axis=1
    )
    repetition_stats = repetition_stats.drop(
        [c for c in repetition_stats.columns if "curve" in c],
        axis=1
    )

    repetition_stats["mean_sensitivity.patient"] = mean_curve_metrics["sensitivity"]
    repetition_stats["mean_specificity.patient"] = mean_curve_metrics["specificity"]
    repetition_stats["mean_sensitivity.img"] = mean_curve_metrics["sensitivity.img"]
    repetition_stats["mean_specificity.img"] = mean_curve_metrics["specificity.img"]

    return stats, repetition_stats, plots


def write_metrics(predictions, output_root, filename_prefix, outcome):
    fold_results, repetition_results, plots = extract_metrics(predictions)
    output_filename_fold = os.path.join(
        output_root,
        "%s_fold.csv" % filename_prefix
    )
    output_filename_rep = os.path.join(
        output_root,
        "%s_repetition.csv" % filename_prefix
    )
    fold_results.transpose().to_csv(output_filename_fold)
    repetition_results.transpose().to_csv(output_filename_rep)

    titlemap = {
        "_cond_diabetes": "",
        "diabetes": "Diabetes",
        "hba1c_high": "HbA1c >= 7.5%",
        "cardiovascular_event": "Cardiovascular Event Status",
        "hypertension": "Hypertension"
    }
    for plot_name, plot in plots.items():
        title = outcome
        for shortform, longform in titlemap.items():
            title = title.replace(shortform, longform)
        plot += ggtitle(title)
        output_filename = os.path.join(
            output_root,
            "%s_%s" % (outcome, plot_name)
        )
        plot.save(filename=output_filename)


def main():
    np.random.seed(config.BOOTSTRAP_SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_predictions", type=str, default=None)
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--subanalysis", type=str, default=None)
    parser.add_argument("--include_all_missing", action="store_true",
                        default=False)
    args = parser.parse_args()

    outcome_root = os.path.join(config.RESULTS_ROOT, args.outcome)
    if args.input_predictions is None:
        input_predictions_fn = os.path.join(outcome_root, "predictions.csv")
    else:
        input_predictions_fn = args.input_predictions

    predictions = pd.read_csv(input_predictions_fn)
    if args.subanalysis is None:
        write_metrics(predictions=predictions,
                      output_root=outcome_root,
                      filename_prefix="prediction_metrics",
                      outcome=args.outcome)
    else:
        input_outcomes_path = os.path.join(config.DATA_ROOT,
                                           "outcome_data.csv")
        outcomes = pd.read_csv(input_outcomes_path).set_index("patient_id")
        hypo_freq = pd.Categorical(
            outcomes["Frequency of symptomatic hypoglycemic episodes"],
            ordered=True,
            categories=["< 1 per month", "1-3 per month",
                        "1-2 per week", ">2 per week"]
        ).codes
        outcomes["Frequency of symptomatic hypoglycemic episodes"] = hypo_freq
        # outcomes["Diabetes Type"] = pd.Categorical(outcomes["Diabetes Type"]).\
        #                             codes
        if args.subanalysis not in outcomes.columns:
            raise ValueError(f"Unknown outcome '{args.subanalysis}'")
        filename_prefix = "prediction_metrics_subanalysis_%s=%s"
        for val in outcomes[args.subanalysis].value_counts().index:
            patient_ids = outcomes.index[outcomes[args.subanalysis] == val]
            if args.include_all_missing:
                patient_ids = list(patient_ids)
                patient_ids += list(outcomes.index[
                    pd.isnull(outcomes[args.subanalysis])
                ])
                try:
                    patient_ids += list(outcomes.index[
                        outcomes[args.subanalysis] < 0
                    ])
                except TypeError:
                    pass
            predictions_sub = predictions[predictions["id"].isin(patient_ids)]
            try:
                write_metrics(
                    predictions=predictions_sub,
                    output_root=outcome_root,
                    filename_prefix=(
                        filename_prefix % (args.subanalysis, val)
                    ),
                    outcome=(
                        "%s (%s: %s)" % (args.outcome,
                                         args.subanalysis,
                                         val)
                    )
                )
            # pylint: disable=bare-except; analysis can fail for any reason
            except:
                print(f"Subanalysis '{args.subanalysis}' = '{val}' failed.")
                traceback.print_exc()
                print("Continuing...")

    print("Done.")

if __name__ == "__main__":
    main()
