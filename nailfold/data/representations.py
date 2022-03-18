"""Code for working with the hidden representations extracted from a NN."""

from typing import Callable, Dict, Any, Optional

import pickle

import numpy as np

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from nailfold import config
from .base import BaseNailfoldDataset


class RepresentationsDataset(Dataset):
    """
    Dataset which handles the image representations (i.e. vectors) extracted
    from the last layer of a predictive neural net.
    """

    def __init__(self,
                 data_root: str,
                 representations: Dict[str, Any],
                 split: str,
                 measure: str,
                 reduce_fn: Callable,
                 measure_transform: Optional[Callable] = None) -> None:
        """
        Args:
            data_root: The root of the dataset.
            representations: The dictionary of the extracted representations
                             and associated metadata.
            split: Which split of the data to use (either train, val, or test).
            measure: Which measure from the annotated images to use.
            reduce_fn: A Callable to use when multiple measures are available
                       for a given patient, for reducing down to a single
                       measurement.
            measure_transform: When given, a Callable to use on the measures
                               during the data loading step.
        """
        super().__init__()

        self.measure = measure
        self.reduce_fn = reduce_fn
        self.representations = representations["results"]["features"]
        self.outcome = representations["outcome"]
        results = representations["results"]
        image_indices = np.where(np.asarray(results["split"]) == split)[0]
        self.base_dataset = BaseNailfoldDataset(data_root)
        self.rep_image_paths = results["image_path"][image_indices]
        self.annot_image_paths = {
            annot_fn: self.base_dataset[sample_idx]["id"]
            for sample_idx in range(len(self.base_dataset))
            for annot_fn in self.base_dataset[sample_idx]["image_annotations"]
            if (self.measure in
                self.base_dataset[sample_idx]["image_annotations"][annot_fn] and
                not np.isnan(reduce_fn(
                    self.base_dataset[sample_idx]["image_annotations"]\
                                     [annot_fn][self.measure]
                ))
            )
        }
        self.image_paths = np.intersect1d(self.rep_image_paths,
                                          list(self.annot_image_paths.keys()))
        self.measure_transform = measure_transform

    def __len__(self):
        """
        The number of representations available.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a representation and corresponding measure.

        Args:
            idx: The index of the representation.
        """
        image_path = self.image_paths[idx]

        # Representation
        rep_idx = np.where(np.asarray(self.rep_image_paths) == image_path)[0]
        representation = self.representations[rep_idx]

        # Annoations
        patient_id = self.annot_image_paths[image_path]
        base_idx = np.where(self.base_dataset.patient_ids == patient_id)[0][0]
        base_sample = self.base_dataset[base_idx]
        annot = base_sample["image_annotations"][image_path]

        # Baseline demographics
        age = base_sample["age"]
        gender = base_sample["gender"]

        sample = {
            "patient_id": patient_id,
            "representation": representation.squeeze(),
            "measure": self.reduce_fn(annot[self.measure]),
            "outcome": base_sample[self.outcome],
            "image_path": image_path,
            "age": age,
            "gender": gender
        }
        if self.measure_transform:
            sample["measure"] = self.measure_transform(sample["measure"])

        return sample

    def compute_statistics(self):
        """
        Compute dataset-level statistics of the associated annotated image
        measures.
        """
        xs = []
        for sample in self:
            xs.append(sample["measure"])

        return np.mean(xs), np.std(xs)


class RepresentationsDataModule(pl.LightningDataModule):
    """
    Data module used for accessing representations.

    Note: In this context, train/val/test correspond to whether the
          representation originated from a sample used for training/validating/
          testing the predictive model. In other words, the split is the same.
    """

    def __init__(self,
                 data_root: str,
                 representations_fn: str,
                 measure: str,
                 reduce_fn: Callable,
                 batch_size: int = 32,
                 normalize: bool = False) -> None:
        """
        Args:
            data_root: The root of the dataset.
            representations_fn: The pickle file containing the computed
                                representations and associated metadata.
            measure: The measure of interest in the annotated images to extract
                     as an outcome.
            reduce_fn: Callable used to reduce multiple measures of interest for
                       a patient to a single measure.
            batch_size: The size of the batch to use.
            normalize: Whether to standardize the annotated image measures.
        """
        super().__init__()

        self.data_root = data_root
        self.representations_fn = representations_fn
        self.measure = measure
        self.reduce_fn = reduce_fn
        self.batch_size = batch_size
        self.normalize = normalize

    def setup(self,
              stage: Optional[str] = None) -> None:
        """
        Set up the underlying datasets, loading the representations into memory.
        """
        representations = pickle.load(open(self.representations_fn, "rb"))

        self.outcome = representations["outcome"]

        self.train_set = RepresentationsDataset(self.data_root,
                                                representations,
                                                "train",
                                                self.measure,
                                                self.reduce_fn)
        m, s = self.train_set.compute_statistics()
        if self.normalize:
            mt = lambda x: (x - m) / s
        else:
            mt = lambda x: x
        self.train_set.measure_transform = mt
        self.val_set = RepresentationsDataset(self.data_root,
                                              representations,
                                              "val",
                                              self.measure,
                                              self.reduce_fn,
                                              measure_transform=mt)
        self.test_set = RepresentationsDataset(self.data_root,
                                               representations,
                                               "test",
                                               self.measure,
                                               self.reduce_fn,
                                               measure_transform=mt)

    def train_dataloader(self) -> DataLoader:
        """
        The dataloader for the train subset.
        """
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        The dataloader for the validation subset.
        """
        return DataLoader(self.val_set,
                          batch_size=config.MAX_REPRESENTATION_BATCH_SIZE)

    def test_dataloader(self) -> DataLoader:
        """
        The dataloader for the validation subset.
        """
        return DataLoader(self.test_set,
                          batch_size=config.MAX_REPRESENTATION_BATCH_SIZE)
