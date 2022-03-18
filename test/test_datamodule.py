import pytest

import torch

import numpy as np

from nailfold import config
from nailfold.data import NailfoldDataModule


def get_datamodule(fold, outcome='diabetes', split_seed=0):
    dm = NailfoldDataModule(binary_covariate_name=outcome,
                            split_seed=split_seed,
                            fold=fold)
    dm.setup()

    return dm


@pytest.fixture
def dm():
    return get_datamodule(fold=0)


@pytest.fixture
def dm_with_missingness():
    return get_datamodule(outcome='hba1c_high', fold=0)


@pytest.fixture
def all_test_idxs():
    '''Returns the test indices from each fold.'''
    all_test_idxs = []
    for fold in range(config.NUM_FOLDS):
        dm = get_datamodule(fold)
        all_test_idxs.append(dm.test_set.base.indices)

    return all_test_idxs


def test_no_test_set_overlap(all_test_idxs):
    '''Ensures that the test sets for each fold do not overlap.'''
    for i in range(len(all_test_idxs)):
        for j in range(len(all_test_idxs)):
            if i != j:
                idxi = all_test_idxs[i]
                idxj = all_test_idxs[j]
                assert len(np.intersect1d(idxi, idxj)) == 0


def test_all_images_tested_once(all_test_idxs):
    '''Ensures that all patients are tested at exactly once.'''
    flat_idx = np.sort(np.concatenate(all_test_idxs))
    assert (flat_idx == np.arange(np.max(flat_idx) + 1)).all()


def test_no_split_overlap():
    '''Ensures that there is no overlap between the train/val/test splits.'''
    for fold in range(config.NUM_FOLDS):
        dm = get_datamodule(fold)
        train_idx = dm.train_set.base.indices
        val_idx = dm.val_set.base.indices
        test_idx = dm.test_set.base.indices

        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(val_idx, test_idx)) == 0


def test_different_split_seeds_give_different_splits():
    '''Different split sheeds should result in different train/val/test splits.
    '''
    for fold in range(config.NUM_FOLDS):
        dm1 = get_datamodule(fold, split_seed=0)
        dm2 = get_datamodule(fold, split_seed=1)

        train_idx1 = dm1.train_set.base.indices
        train_idx2 = dm2.train_set.base.indices
        num_shared = len(np.intersect1d(train_idx1, train_idx2))
        assert num_shared < min([len(train_idx1), len(train_idx2)])

        val_idx1 = dm1.val_set.base.indices
        val_idx2 = dm2.val_set.base.indices
        num_shared = len(np.intersect1d(val_idx1, val_idx2))
        assert num_shared < min([len(val_idx1), len(val_idx2)])

        test_idx1 = dm1.test_set.base.indices
        test_idx2 = dm2.test_set.base.indices
        num_shared = len(np.intersect1d(test_idx1, test_idx2))
        assert num_shared < min([len(test_idx1), len(test_idx2)])


def test_train_dataloader_uses_train_set(dm):
    train_dl = dm.train_dataloader()
    assert train_dl.dataset == dm.train_set


def test_val_dataloader_uses_val_set(dm):
    val_dl = dm.val_dataloader()
    assert val_dl.dataset == dm.val_set


def test_test_dataloader_uses_test_set(dm):
    test_dl = dm.test_dataloader()
    assert test_dl.dataset == dm.test_set


def test_image_batch_dimensions_are_correct(dm):
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))

    b = config.MAX_BATCH_SIZE
    c = 3
    h, w = config.INPUT_IMAGE_SIZE

    assert batch['image'].shape == torch.Size([b, c, h, w])


def test_dm_with_missingness_returns_nonmissing_patients(dm_with_missingness):
    outcome_name = dm_with_missingness.outcome_name
    for patient in dm_with_missingness.train_set.samples:
        assert not np.isnan(patient['outcome'])
    for patient in dm_with_missingness.val_set.samples:
        assert not np.isnan(patient['outcome'])
    for patient in dm_with_missingness.test_set.samples:
        assert not np.isnan(patient['outcome'])
