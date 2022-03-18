"""The main DataModule for supervised classification."""

from typing import Tuple, Optional

from sklearn.model_selection import StratifiedKFold, train_test_split

import numpy as np

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

from nailfold import config
from nailfold.transforms import (augmentation_fn as default_augmentation_fn,
                                 normalization_fn as imagenet_normalization_fn)
from .base import BaseNailfoldDataset
from .supervised import SupervisedNailfoldDataset


class NailfoldDataModule(pl.LightningDataModule):
    """The PyTorch Lightning DataModule wrapper around the Nailfold dataset.

    The assumptions one can make about this data module are:
        (1) k-fold cross-validation
        (2) tune set split from train set
        (3) seed will reproduce the patient split exactly
    """

    def __init__(self,
                 fold: int,
                 batch_size: int = config.MAX_BATCH_SIZE,
                 num_workers: int = config.NUM_DATALOADER_WORKERS,
                 val_prop: float = config.VAL_SET_PROPORTION,
                 continuous_covariate_name: str = None,
                 binary_covariate_name: str = None,
                 condition_on: str = None,
                 data_root: str = config.DATA_ROOT,
                 image_size: Tuple[int, int] = config.INPUT_IMAGE_SIZE,
                 augmentation_fn: nn.Module = default_augmentation_fn,
                 normalization_fn: nn.Module = imagenet_normalization_fn,
                 split_seed: int = 0,
                 verbose: bool = False) -> None:
        """Create the DataModule.

        Parameters:
            fold : int
                The index of the held-out fold, from 0 to
                `config.NUM_FOLDS - 1`.
            batch_size : int
                The batch size for the dataloaders (incl. train, val, and
                test).
            num_workers : int
                The number of workers used for each dataloader.
            val_prop : float
                The proportion of the train fold to split off as the validation
                set.
            continuous_covariate_name : str
                The name of the continuous covariate to fit the model to. This
                or `binary_covariate_name` must not be None, but both cannot be
                provided.
            binary_covariate_name : str
                The name of the bianry covariate to fit the model to. This or
                `continuous_covariate_name` must not be None, but both cannot
                be provided.
            condition_on : str (optional)
                The covariate to condition on. If None, ignored.
            data_root : str
                The root path of the data.
            image_size : Tuple[int, int]
                The dimension of the images to resize to.
            augmentation_fn : Callable
                The function used to augment each image during training.
            normalization_fn : Callable
                The function used to normalize each input sample during
                training.
            split_seed : int
                The seed to use for splitting. If set, the patient split should
                be completely the same for the same set of input parameters.
            verbose : bool
                Whether to output summary information to the terminal.
        """
        super().__init__()

        self.verbose = verbose

        # Outcome
        if continuous_covariate_name is None:
            if binary_covariate_name is None:
                raise ValueError("Must provide a covariate to use as outcome!")

            self.mode = "binary"
            self.outcome_name = binary_covariate_name
        else:
            self.mode = "continuous"
            self.outcome_name = continuous_covariate_name

        if condition_on:
            self.conditional_cov_name: Optional[str] = condition_on
        else:
            self.conditional_cov_name = None

        # Train/val/test split meta
        self.fold = fold
        self.val_prop = val_prop
        self.split_seed = split_seed

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Dataset
        self.base = BaseNailfoldDataset(data_root)
        self.eval_outcome_transform = self.base.eval_outcome_transform_for(
            self.outcome_name
        )

        # Transforms
        self.image_size = image_size
        self.train_transform = nn.Sequential(
            augmentation_fn,
            normalization_fn
        )
        self.test_transform = normalization_fn

    def setup(self, stage=None):
        """Split the data and build the data subsets.

        Internally, uses scikit-learn"s stratified data splitting routines to
        first do the k-fold split. Then, uses stratified shuffled splitting to
        split off the validation set from the training set."""
        if self.mode == "continuous":
            outcome = [~np.isnan(sample[self.outcome_name])
                       for sample in self.base]
        elif self.mode == "binary":
            outcome = [sample[self.outcome_name]
                       if ~np.isnan(sample[self.outcome_name]) else -9
                       for sample in self.base]

        splitter = StratifiedKFold(n_splits=config.NUM_FOLDS,
                                   shuffle=True,
                                   random_state=self.split_seed)
        splits = list(splitter.split(torch.zeros(len(outcome)), outcome))
        train_val_idx, test_idx = splits[self.fold]
        train_val_idx = torch.from_numpy(train_val_idx)

        train_val_outcome = [outcome[i] for i in train_val_idx]
        train_idx_idx, val_idx_idx = train_test_split(
            torch.arange(len(train_val_outcome)),
            test_size=self.val_prop,
            random_state=(self.split_seed * config.NUM_FOLDS + self.fold),
            stratify=train_val_outcome
        )
        train_idx = train_val_idx[train_idx_idx]
        val_idx = train_val_idx[val_idx_idx]

        # Visualize the split
        visual = np.array(list(" " * len(self.base)))
        visual[train_idx] = "_"
        visual[val_idx] = "V"
        visual[test_idx] = "T"
        if self.verbose:
            print("Split: %s" % "".join(visual))

        self.train_set = SupervisedNailfoldDataset(
            Subset(self.base, train_idx),
            self.mode,
            self.outcome_name,
            self.conditional_cov_name,
            transform=self.train_transform,
            image_size=self.image_size,
            verbose=self.verbose
        )
        self.train_set.summary("training")

        self.val_set = SupervisedNailfoldDataset(Subset(self.base, val_idx),
                                                 self.mode,
                                                 self.outcome_name,
                                                 self.conditional_cov_name,
                                                 transform=self.test_transform,
                                                 image_size=self.image_size,
                                                 verbose=self.verbose)
        self.val_set.summary("validation")

        self.test_set = SupervisedNailfoldDataset(
            Subset(self.base, test_idx),
            self.mode,
            self.outcome_name,
            self.conditional_cov_name,
            transform=self.test_transform,
            image_size=self.image_size,
            verbose=self.verbose
        )
        self.test_set.summary("testing")

    def train_dataloader(self) -> DataLoader:
        """Get the DataLoader for the training set."""
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Get the DataLoader for the validation set."""
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Get the DataLoader for the test set."""
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
