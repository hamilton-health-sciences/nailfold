"""Evaluation metric utilities."""

from typing import Callable

import numpy as np

from sklearn.metrics import (roc_curve as roc_curve_base,
                             precision_recall_curve as pr_curve_base,
                             classification_report)

from nailfold import config


baseline_x = np.linspace(0., 1., 251)

def roc_curve(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """
    Compute the ROC curve at a fixed set of interpolation points so that they
    can be combined after cross-validation.

    Args:
        y: The targets.
        yhat: The predictions.

    Returns:
        tpr_interp: The true positive rate at the interpolated false positive
                    rate points.
    """
    fpr, tpr, _ = roc_curve_base(y, yhat)
    tpr_interp = np.interp(baseline_x, fpr, tpr)
    # Set left-most point to zero (for the rest of the points we want the
    # default behaviour of taking the max)
    tpr_interp[0] = 0.

    return tpr_interp


def pr_curve(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """
    Compute the PR curve at a fixed set of interpolation points so that they
    can be combined after cross-validation.

    Args:
        y: The targets.
        yhat: The predictions.

    Returns:
        precision_interp: The precision at the interpolated recall points.
    """
    precision, recall, _ = pr_curve_base(y, yhat)
    # Flip so in ascending order - does not affect interpolation.
    precision_interp = np.interp(baseline_x,
                                 np.flip(recall),
                                 np.flip(precision))

    return precision_interp


def youden_index(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Find the optimal classification threshold by maximizing Youden's index.

    Parameters:
        y : array-like
            Binary labels.
        yhat : array-like
            Predicted score.

    Returns:
        best_thresh : float
            The optimal threshold for the scores.
    """
    fpr, tpr, thresh = roc_curve_base(y, yhat)
    best = np.argmax(tpr - fpr)
    best_thresh = thresh[best]

    return best_thresh


def specificity(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Compute the specificity of a set of predictions at Youden's index.

    Parameters:
        y : array-like
            Binary labels.
        yhat : array-like
            Predicted score.

    Returns:
        spec : float
            The specificity at Youden's index.
    """
    thresh = youden_index(y, yhat)
    preds = (yhat >= thresh).astype(int)
    y = y.astype(int)
    report = classification_report(y, preds, output_dict=True)

    if "0" not in report:
        spec = 0.
    else:
        spec = report["0"]["recall"]

    return spec


def sensitivity(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Compute the sensitivity of a set of predictions at Youden's index.

    Parameters:
        y : array-like
            Binary labels.
        yhat : array-like
            Predicted score.

    Returns:
        sens : float
            The specificity at Youden's index.
    """
    thresh = youden_index(y, yhat)
    preds = (yhat >= thresh).astype(int)
    y = y.astype(int)
    report = classification_report(y, preds, output_dict=True)

    if "1" not in report:
        sens = 0.
    else:
        sens = report["1"]["recall"]

    return sens


def compute_bootstraps(yhat: np.ndarray,
                       y: np.ndarray,
                       stat_func: Callable,
                       num_samples=config.NUM_BOOTSTRAP_SAMPLES) -> np.ndarray:
    """
    Bootstrap a statistic.

    Args:
        yhat: Predicted scores.
        y: Binary outcomes.
        stat_func: The function used to compute the statistic.
        num_samples: The number of samples to bootstrap when computing the
                     statistic.

    Returns:
        stats: An array of the bootstrapped statistics.
    """
    stats = []
    while len(stats) < num_samples:
        idx = np.random.choice(len(y), size=len(y), replace=True)
        try:
            stat = stat_func(y[idx], yhat[idx])
            if np.sum([np.isnan(stat)]) == 0:
                stats.append(stat)
        # pylint: disable=bare-except; stat_func is arbitrary
        except:
            pass
    stats = np.asarray(stats)

    return stats


# pylint: disable=unused-argument; needs same signature as other metrics
def positive_class_frequency(y: np.ndarray,
                             yhat: np.ndarray) -> float:
    """
    Compute the frequency of the positive class.

    Args:
        y: Targets.
        yhat: Ignored.

    Returns:
        freq: The frequency of the positive class.
    """
    freq = (y == 1.).sum() / len(y)

    return freq
