"""Wrapper around the base nailfold dataset which provides single images."""

from typing import Callable, Optional, Union, Dict, Tuple, Any

import cv2

from torch.utils.data import Dataset, Subset

import kornia

import numpy as np

from .base import BaseNailfoldDataset


class SupervisedNailfoldDataset(Dataset):
    """Wrapper around the base dataset for supervised learning problems with a
    single outcome."""

    def __init__(self,
                 base: Union[BaseNailfoldDataset, Subset],
                 mode: str,
                 outcome_name: str,
                 conditional_cov_name: Optional[str],
                 image_size: Tuple[int, int],
                 transform: Callable,
                 verbose: bool = False) -> None:
        """Build the supervised dataset.

        Parameters:
            base : torch.utils.data.Dataset
                The base nailfold dataset, or a split thereof.
            mode : str
                Either "binary" or "continuous" depending on outcome type.
            outcome_name : str
                The name of the outcome to pair with the images.
            conditional_cov_name : str (optional)
                The name of the covariate to condition on. If None, ignored.
            image_size : list-like of int
                The dimensions of the images to resize to.
            transform : Callable
                The transform to apply, separate from resizing.
            verbose : bool
                Whether to output summary information to the terminal. Useful
                for debugging.
        """
        self.base = base
        self.mode = mode
        self.outcome_name = outcome_name
        self.conditional_cov_name = conditional_cov_name
        self.image_size = image_size
        self.transform = transform
        self.verbose = verbose

        self.samples = []
        for patient_idx in range(len(self.base)):
            patient = self.base[patient_idx]
            if not np.isnan(patient[self.outcome_name]):
                for image_filepath in patient["image_filepaths"]:
                    sample = {
                        "id": patient["id"],
                        "outcome": patient[self.outcome_name],
                        "image_filepath": image_filepath
                    }
                    if self.conditional_cov_name is not None:
                        cond_cov = patient[self.conditional_cov_name]
                        sample["conditional_cov"] = cond_cov
                    self.samples.append(sample)

    def __len__(self) -> int:
        """The number of images in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get the image at index `idx`."""
        sample = self.samples[idx]

        x = cv2.imread(sample["image_filepath"])
        x = x.astype(np.float32) / 255.
        x = kornia.image_to_tensor(x)
        x = kornia.geometry.transform.resize(
            x.unsqueeze(0),
            size=self.image_size
        ).squeeze(0)
        if self.transform:
            x = self.transform(x)
        x = x.squeeze()

        sample["image"] = x
        sample["outcome"] = sample["outcome"].astype(np.float32)
        if "conditional_cov" in sample:
            cond_cov = sample["conditional_cov"]
            sample["conditional_cov"] = cond_cov.astype(np.float32)

        return sample

    def summary(self, subset_name="(all)") -> None:
        """Summarize the dataset to the command-line."""
        if self.verbose:
            print("Subset: %s" % subset_name)
            num_patients = len(np.unique([x["id"] for x in self.samples]))
            num_images = len(self.samples)
            print("  # of patients: %d" % num_patients)
            print("  # of images:   %d" % num_images)
            if self.mode == "binary":
                num_cases = np.sum([
                    sample["outcome"] is True for sample in self.samples
                ])
                num_controls = np.sum([
                    sample["outcome"] is False for sample in self.samples
                ])

                print("  # of case images:    %d" % num_cases)
                print("  # of control images: %d" % num_controls)

    @property
    def case_weight(self) -> float:
        """Compute the case weight for a weighted loss function for a given
        outcome based on the dataset.

        Note: this should only be called on the training set."""
        # pylint: disable=singleton-comparison; need independent nan check
        num_cases = np.sum([
            sample["outcome"] == True for sample in self.samples
        ]).astype(np.float32).item()
        # pylint: disable=singleton-comparison; need independent nan check
        num_controls = np.sum([
            sample["outcome"] == False for sample in self.samples
        ]).astype(np.float32).item()

        case_weight = num_controls / num_cases
        print("Case weight: %f" % case_weight)

        return case_weight
