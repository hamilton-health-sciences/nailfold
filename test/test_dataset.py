import os

import pytest

import numpy as np

from nailfold import config
from nailfold.data.base import BaseNailfoldDataset


@pytest.fixture
def dataset():
    return BaseNailfoldDataset(config.DATA_ROOT)


def test_binary_outcomes_are_binarized(dataset):
    '''Ensure that the pre-processing of the tabular data is correct.'''
    binary_outcomes = ['diabetes', 'hba1c_high', 'cardiovascular_event',
                       'hypertension', 'retinopathy']
    for patient in iter(dataset):
        for binary_outcome in binary_outcomes:
            y = patient[binary_outcome]
            # Allow outcome to be missing for a given patient
            if not np.isnan(y):
                # Otherwise, it needs to be a boolean value
                assert y in [False, True]


def test_image_filepaths_are_correctly_assigned(dataset):
    '''Ensure that image files are assigned to the correct patient.'''
    for patient in iter(dataset):
        for image_filepath in patient['image_filepaths']:
            image_patient_id = int(image_filepath.split('/')[-3].split('_')[0])
            assert image_patient_id == patient['id']


def test_image_filepaths_are_unique(dataset):
    '''Ensure that each image file is present only once.'''
    for patient in iter(dataset):
        num_unique = len(set(patient['image_filepaths']))
        num_paths = len(patient['image_filepaths'])
        assert num_unique == num_paths


def test_patient_ids_are_unique(dataset):
    assert len(dataset.patient_ids) == len(np.unique(dataset.patient_ids))
