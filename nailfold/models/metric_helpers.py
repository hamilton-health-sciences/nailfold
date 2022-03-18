"""Helpers for building the evaluation metrics for the model."""

import torch
from torch import nn

import pytorch_lightning as pl

from nailfold.metrics import PatientAggregated


def build_binary_metrics():
    """Build the metrics for a binary outcome.

    Included metrics are image- and patient-level accuracy and AUPR, the image-
    level ROC curve, and the patient-level AUROC."""
    metrics = nn.ModuleDict({
        "img_accuracy": pl.metrics.Accuracy(threshold=0.,
                                            compute_on_step=False),
        "img_roc": pl.metrics.ROC(compute_on_step=False),
        "img_aupr": pl.metrics.AveragePrecision(
            compute_on_step=False
        ),
        "patient_accuracy": PatientAggregated(
            pl.metrics.functional.classification.accuracy,
            compute_on_step=False,
            pred_transform=lambda x: (x > 0).float()
        ),
        "patient_auroc": PatientAggregated(
            pl.metrics.functional.classification.auroc,
            compute_on_step=False,
            pred_transform=torch.sigmoid
        ),
        "patient_aupr": PatientAggregated(
            pl.metrics.functional.average_precision,
            compute_on_step=False,
            pred_transform=torch.sigmoid
        )
    })

    return metrics


def build_continuous_metrics():
    """Build the metrics for a continuous outcome.

    Included metrics are mean-squared error (mse), mean absolute error (mae),
    and variance explained (r2)."""
    metrics = nn.ModuleDict({
        "img_mse": pl.metrics.MeanSquaredError(compute_on_step=False),
        "img_mae": pl.metrics.MeanAbsoluteError(compute_on_step=False),
        "img_r2": pl.metrics.ExplainedVariance(compute_on_step=False),
        "patient_mse": PatientAggregated(
            pl.metrics.functional.mean_squared_error,
            compute_on_step=False
        ),
        "patient_mae": PatientAggregated(
            pl.metrics.functional.mean_absolute_error,
            compute_on_step=False
        ),
        "patient_r2": PatientAggregated(
            pl.metrics.functional.explained_variance,
            compute_on_step=False
        )
    })

    return metrics
