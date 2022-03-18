"""Compute the Table 1 (summary) statistics for the dataset."""

import pandas as pd

from nailfold import config
from nailfold.data.base import BaseNailfoldDataset


def main():
    data = {
        "id": [],
        "age": [],
        "gender": [],
        "race": [],
        "diabetes_type": [],
        "hypoglycemic_episode_frequency": [],
        "diabetes": [],
        "hba1c_high": [],
        "cve": [],
        "hypertension": [],
        "retinopathy": [],
        "albuminuria": [],
        "num_images": []
    }
    dataset = BaseNailfoldDataset(config.DATA_ROOT)
    for patient in dataset:
        data["id"].append(patient["id"])
        data["num_images"].append(len(patient["image_filepaths"]))
    for patient_id in data["id"]:
        patient_info = dataset.raw_patient_info.loc[patient_id]
        data["age"].append(patient_info["Age"])
        data["gender"].append(patient_info["Gender"])
        data["race"].append(patient_info["Race"])
        data["diabetes_type"].append(patient_info["Diabetes Type"])
        data["hypoglycemic_episode_frequency"].append(
            patient_info["Frequency of symptomatic hypoglycemic episodes"]
        )

        patient_info = dataset.patient_info.loc[patient_id]
        data["diabetes"].append(patient_info["diabetes"])
        data["hba1c_high"].append(patient_info["hba1c_high"])
        data["cve"].append(patient_info["cardiovascular_event"])
        data["hypertension"].append(patient_info["hypertension"])
        data["retinopathy"].append(patient_info["retinopathy"])
        data["albuminuria"].append(patient_info["albuminuria"])
    df = pd.DataFrame(data).set_index("id")

    print("Mean Age (IQR): %.1f (%.1f - %.1f)" %
          tuple(list(df["age"].describe()[["mean", "25%", "75%"]])))
    print("Race:")
    print(df["race"].value_counts())
    print("Hypoglycemic episode frequency:")
    print(df["hypoglycemic_episode_frequency"].value_counts())
    bin_vars = ["gender", "diabetes_type", "diabetes", "hba1c_high", "cve",
                "hypertension", "retinopathy", "albuminuria"]
    for bin_var in bin_vars:
        df[bin_var] = pd.Categorical(df[bin_var])

    df_sub = df.loc[df["diabetes"]]

    print("[d] Mean Age (IQR): %.1f (%.1f - %.1f)" %
          tuple(list(df_sub["age"].describe()[["mean", "25%", "75%"]])))
    print("[d] Race:")
    print(df_sub["race"].value_counts())

    num_patients = df.shape[0]
    for bin_var in bin_vars:
        freq = df[bin_var].describe()["freq"]
        count = df[bin_var].describe()["count"]
        missingness = 1. - (count / num_patients)
        prop = freq / count
        top = df[bin_var].describe()["top"]

        print("%s [%s] (count / total, proportion, proportion missing): "
              "%d / %d, %.3f, %.3f" % (bin_var, top, freq, count, prop,
                                       missingness))
        freq = df_sub[bin_var].describe()["freq"]
        count = df_sub[bin_var].describe()["count"]
        missingness = 1. - (count / num_patients)
        prop = freq / count
        top = df_sub[bin_var].describe()["top"]

        print("[d] %s [%s] (count / total, proportion, proportion missing): "
              "%d / %d, %.3f, %.3f" % (bin_var, top, freq, count, prop,
                                       missingness))
    for bin_var in bin_vars:
        top = df[bin_var].describe()["top"]
        top_freq = df[bin_var].describe()["freq"]
        other = df.loc[df[bin_var] != top, bin_var].describe()["top"]
        other_freq = df.loc[df[bin_var] != top, bin_var].describe()["freq"]
        num_img_top = df[df[bin_var] == top]["num_images"].sum()
        num_img_other = df[df[bin_var] == other]["num_images"].sum()

        print("%s:" % bin_var)
        print("  %s: %d patients (%d images)" % (top, top_freq, num_img_top))
        print("  %s: %d patients (%d images)" % (other, other_freq,
                                                 num_img_other))


if __name__ == "__main__":
    main()
