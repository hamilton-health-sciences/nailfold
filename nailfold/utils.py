"""Various utility functions."""

from typing import Any, List, Dict

import os

import numpy as np

from ray.tune import Analysis

from . import config
from .data.module import NailfoldDataModule
from .models import TorchvisionWrapper


def get_annotated_supervised_samples_for_patient(
    dm: NailfoldDataModule,
    patient_id: int
) -> List[Dict[str, Any]]:
    """
    For a given patient, get the images for the patient which have associated
    image annotations.

    Args:
        dm: The data module of interest.
        patient_id: The ID of the patient of interest.

    Returns:
        samples: The list of samples (images) which have associated annotations.
    """
    # Identify which images for the patient have annotations
    base_dataset_idx = np.where(
        dm.test_set.base.dataset.patient_ids == patient_id
    )[0][0]
    sample = dm.test_set.base.dataset[base_dataset_idx]

    # Randomly select an image filepath with annotations
    image_paths = list(sample["image_annotations"].keys())

    samples = []
    for subset in [dm.train_set, dm.val_set, dm.test_set]:
        # Identify in the index in the supervised dataset
        supervised_dataset_idx = np.where([
            any(image_path in sample["image_filepath"]
                for image_path in image_paths)
            for sample in subset.samples
        ])[0]
        if len(supervised_dataset_idx) > 0:
            for idx in supervised_dataset_idx:
                # Load the sample
                sample = subset[idx]
                samples.append(sample)

    return samples

def get_best_model(logs_root: str,
                   outcome: str,
                   repetition: int,
                   fold: int) -> TorchvisionWrapper:
    """
    For a given outcome, repetition, and fold, get the best-performing model
    according to validation set performance.

    Args:
        logs_root: The root of the output Ray logs.
        outcome: The outcome of interest.
        repetition: The repetition index.
        fold: The fold index.

    Returns:
        model: The optimal model according to validation set performance.
    """
    experiment_path = os.path.join(
        logs_root, "%s_repetition=%d_fold=%d" % (outcome, repetition, fold)
    )
    analysis = Analysis(experiment_path)
    df = analysis.dataframe()
    df_completed = df[df["iterations_since_restore"] == config.NUM_EPOCHS]
    best_by_last = df_completed.loc[df_completed["val_img_auroc"].idxmax()]
    checkpoints_dir = os.path.join(best_by_last.logdir, "checkpoints")
    print("Best model for rep %d, fold %d: %s" %
          (repetition, fold, best_by_last.logdir))
    best_checkpoint = "last.ckpt"
    checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
    model = TorchvisionWrapper.load_from_checkpoint(checkpoint_path)

    return model
