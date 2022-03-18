"""Base nailfold dataset, which provides all image paths for a sample."""

from typing import Optional, Callable, Dict, Any

import os

import pandas as pd

import torch
from torch.utils.data import Dataset

import numpy as np


class BaseNailfoldDataset(Dataset):
    """Dataset that represents all data for the project.

    Specifically, selected outcome data is extracted, along with the
    corresponding image filenames."""

    def __init__(self, dataset_root: str) -> None:
        """Initialize the base dataset.

        Parameters:
            dataset_root : str
                The root of the nailfold dataset, which contains the Excel
                spreadsheet with the metadata and labels and the images.
        """
        super().__init__()

        self.dataset_root = dataset_root

        self.available = [
            x for x in os.listdir(self.dataset_root)
            if "xlsx" not in x and "docx" not in x and "csv" not in x
        ]
        self.patient_ids = np.asarray([
            int(dirname.split("_")[0]) for dirname in self.available
        ])

        self.patient_info = self.process_patient_info()
        self.metadata = self.process_metadata()

    def process_patient_info(self) -> pd.DataFrame:
        """Process the outcome data available in the Excel file.

        Largely based on earlier work with K. Roth & M. Ghassemi."""
        patient_info_fn = os.path.join(self.dataset_root, "outcome_data.csv")

        pinfo = pd.read_csv(patient_info_fn).set_index("patient_id")
        self.raw_patient_info = pinfo
        pinfo = pinfo[["Diabetes Diagnosis", "Cardiovascular Event Status",
                       "Laser therapy for retinopathy at any time",
                       "VEGF inhibitor injection at any time",
                       "History of vitrectomy",
                       "Urine albumine:creatinine ratio", "Hemoglobin A1c (%)",
                       "Hypertension", "Age", "Gender"]].copy()
        pinfo.columns = ["diabetes", "cardiovascular_event", "laser_therapy",
                         "vegf_inhibitor", "vitrectomy",
                         "albumin_creatinine_ratio", "hba1c", "hypertension",
                         "age", "gender"]
        pinfo["diabetes"] = (pinfo["diabetes"] == "Yes")
        pinfo["cardiovascular_event"] = (
            pinfo["cardiovascular_event"] == "Yes"
        )
        pinfo["retinopathy"] = (
            (pinfo["laser_therapy"] == "Yes") |
            (pinfo["vegf_inhibitor"] == "Yes") |
            (pinfo["vitrectomy"] == "Yes")
        )
        pinfo["hypertension"] = (pinfo["hypertension"] == "Yes")
        pinfo = pinfo.drop(
            ["laser_therapy", "vegf_inhibitor", "vitrectomy"],
            axis=1
        )

        # Contains zero values, so a small offset (not so small as to skew the
        # dataset heavily) is used before log-transforming.
        pinfo["log_acr"] = np.log(pinfo["albumin_creatinine_ratio"] + 1e-2)
        # Here, we use an offset of 6% because that is the cutoff between non-
        # diabetes and pre-diabetes
        pinfo["hba1c_norm"] = pinfo["hba1c"] - 6.
        # We also create binarized versions of HbA1c ("high HbA1c") for use in a
        # classification context.
        pinfo["hba1c_high"] = (pinfo["hba1c"] >= 7.5)
        pinfo["hba1c_moderatehigh"] = (pinfo["hba1c"] >= 6.5)
        pinfo.loc[pd.isnull(pinfo["hba1c"]), "hba1c_high"] = np.nan
        pinfo.loc[pd.isnull(pinfo["hba1c"]), "hba1c_moderatehigh"] = np.nan
        # Similarly, for albuminuria.
        pinfo["albuminuria"] = (pinfo["albumin_creatinine_ratio"] > 3)

        # Ensure coding of baseline demos
        pinfo["gender"] = (pinfo["gender"] == "Male").astype(int).astype(float)

        return pinfo

    def process_metadata(self) -> pd.DataFrame:
        metadata_fn = os.path.join(self.dataset_root, "metadata.csv")
        metadata = pd.read_csv(metadata_fn).set_index("patient_id")

        return metadata

    def __len__(self) -> int:
        """Return the number of available samples, based on the image
        directory."""
        return len(self.available)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get the patient at a given index."""
        patient_id = self.patient_ids[idx]

        # Outcome information
        patient_info = self.patient_info.loc[patient_id].to_dict()
        patient_info["id"] = patient_id

        # Image filenames
        image_meta = self.metadata.loc[patient_id]
        patient_info["image_filepaths"] = list(
            image_meta["path"].apply(
                lambda relpath: os.path.join(self.dataset_root, relpath)
            )
        )

        # Extract image measurements
        patient_info["image_annotations"] = {}
        micrometers = b"\xc3\x82\xc2\xb5m".decode("utf-8")
        for _, image in image_meta.iterrows():
            img_annot = {}

            # Extract numbers of measurements in micrometers
            def is_micrometer_col(i, img=image) -> bool:
                return img["Distance %d_unit" % i] == micrometers
            idx = list(filter(is_micrometer_col, range(1, 14)))
            # If there is at least one measurement in micrometers, then there
            # should be 3 each of lengths and widths. Sort them and figure it
            # out.
            if len(idx) > 0:
                measures = np.asarray([
                    image["Distance %d" % i] for i in idx
                ])
                sorted_measures = np.sort(measures)
                # split_idx = np.argmax(np.diff(sorted_measures)) + 1
                split_idx = 3
                widths = sorted_measures[:split_idx]
                lengths = sorted_measures[split_idx:]
                img_annot["widths"] = widths
                img_annot["lengths"] = lengths
            # Also extract the capillary counts.
            if ~pd.isnull(image["Capillary"]):
                img_annot["capillary_count"] = image["Capillary"]

            if img_annot:
                patient_info["image_annotations"][image["path"]] = img_annot

        return patient_info

    def eval_outcome_transform_for(self, outcome_name) -> Optional[Callable]:
        """Get the Callable for a given outcome transformation for evaluation.
        """
        f: Optional[Callable] = None

        if outcome_name == "log_acr":
            f = lambda y: torch.exp(y) - 1e-2
        elif outcome_name == "hba1c_norm":
            f = lambda y: y + 6.

        return f
