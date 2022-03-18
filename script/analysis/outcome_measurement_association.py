"""Analyze the relationship between outcomes and measurements."""

import argparse

import pandas as pd

import numpy as np

from scipy.stats import mannwhitneyu

from nailfold import config
from nailfold.data.base import BaseNailfoldDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--measure", type=str, required=True)
    args = parser.parse_args()

    # What to do when multiple measurements are available for the same image
    if args.measure == "capillary_count":
        agg_func = lambda x: x  # pylint: disable=unnecessary-lambda; dummy func
    else:
        agg_func = np.nanmedian

    data = {
        "patient_id": [],
        "outcome": [],
        "image_path": [],
        "measure": []
    }
    ds = BaseNailfoldDataset(dataset_root=config.DATA_ROOT)
    for sample in ds:
        for img_path, measures in sample["image_annotations"].items():
            if args.measure in measures:
                data["patient_id"].append(sample["id"])
                data["outcome"].append(sample[args.outcome])
                data["image_path"].append(img_path)
                data["measure"].append(agg_func(measures[args.measure]))
    df = pd.DataFrame(data).dropna()
    df["outcome"] = df["outcome"].astype(int).astype(bool)
    df_patient = df.groupby("patient_id").mean()

    img_counts = df.groupby("patient_id")["image_path"].count()
    mean = img_counts.mean()
    lower, upper = np.quantile(img_counts, [0.25, 0.75])
    print("Mean number of images / patient: ", mean, lower, upper)

    # Perform M-W U test at image level
    x = df["measure"][df["outcome"]]
    y = df["measure"][~df["outcome"]]
    result = mannwhitneyu(x, y, alternative="two-sided")
    print("Mann-Whitney U (image level): ", result)

    # Perform M-W U test at patient level
    x = df_patient["measure"][df_patient["outcome"]]
    y = df_patient["measure"][~df_patient["outcome"]]
    result = mannwhitneyu(x, y, alternative="two-sided")
    print("Mann-Whitney U (patient level): ", result)

if __name__ == "__main__":
    main()
