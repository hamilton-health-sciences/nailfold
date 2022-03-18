"""Pre-process the data."""

import argparse

import os

from functools import partial

from tqdm import tqdm

import numpy as np

import cv2

import pandas as pd


image_cache = {}


def cached_imread(fp):
    if fp not in image_cache:
        image_cache[fp] = cv2.imread(fp)

    return image_cache[fp]


def clear_image_cache():
    for k in list(image_cache.keys()):
        del image_cache[k]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--enhance", action="store_true", default=False)
    parser.add_argument("--output_root", type=str, required=True)

    args = parser.parse_args()

    if args.enhance:
        print("Enhancing images will create new files.")
    else:
        print("Not enhancing images will create symlinks to the original "
              "files.")

    outcome_data_fn = "Nailfold Project - Patient Data.xlsx"
    outcome_data = pd.read_excel(os.path.join(args.data_root, outcome_data_fn))
    outcome_data = outcome_data.rename({"Number": "patient_id"}, axis=1)
    outcome_data["patient_id"] = outcome_data["patient_id"].astype(int)
    outcome_data = outcome_data.set_index("patient_id")
    outcome_data.to_csv(os.path.join(args.output_root, "outcome_data.csv"))

    samples = []
    walk = list(os.walk(args.data_root))
    for patient_root, _, filenames in tqdm(walk):
        patient_measurements = {}
        for filename in filenames:
            dirpath = os.path.relpath(patient_root, args.data_root)
            patient_id = dirpath.split("_")[0]
            if "jpg" in filename:
                # For patient 37, the outer directory is named for patient 38,
                # the inner files are labelled 37, so we need to account for
                # this.
                patient_id_in_fn = ((patient_id in filename) or
                                    (str(int(patient_id) - 1) in filename))
                if patient_id_in_fn and "scored" in filename:
                    # Attempt to process the measurements file
                    base_fn = os.path.splitext(filename)[0]
                    measurements_fn = "%s.txt" % base_fn
                    measurements_fullpath = os.path.join(args.data_root,
                                                         dirpath,
                                                         measurements_fn)
                    measurements_df = pd.read_csv(
                        measurements_fullpath,
                        sep="\t",
                        skiprows=12,
                        encoding="latin1",
                        index_col=False
                    )

                    # Identify which image has been annotated
                    data_root = args.data_root
                    img = cv2.imread(
                        os.path.join(data_root, dirpath, filename)
                    )
                    def pixelwise_img_diff(filepath, target):
                        return np.abs(cached_imread(filepath) - target).sum()
                    annotated_image_filepath = min(
                        filter(
                            os.path.exists,
                            map(
                                partial(os.path.join, data_root, dirpath),
                                filter(
                                    lambda f: "jpg" in f and "scored" not in f,
                                    filenames
                                )
                            )
                        ),
                        key=partial(pixelwise_img_diff, target=img)
                    )

                    # Associate the extracted measurements with that image
                    annotated_fn = os.path.basename(annotated_image_filepath)
                    patient_measurements[annotated_fn] = (filename,
                                                          measurements_df)

        for filename in filenames:
            dirpath = os.path.relpath(patient_root, args.data_root)
            patient_id = dirpath.split("_")[0]
            if "jpg" in filename:
                if "scored" not in filename:
                    try:
                        # Add file to metadata
                        hand_finger, id_ext = filename.split("-")
                        hand = hand_finger[0]
                        finger = int(hand_finger[1:])
                        image_id, _ = os.path.splitext(id_ext)

                        sample = {
                            "patient_id": int(patient_id),
                            "hand": hand,
                            "finger": finger,
                            "image_id": image_id,
                            "path": os.path.join(dirpath, filename)
                        }

                        # Check if there were associated measurements
                        if filename in patient_measurements:
                            fn = filename
                            for _, s in patient_measurements[fn][1].iterrows():
                                sample[s.Tool] = s.Value
                                sample["%s_unit" % s.Tool] = s.Unit

                        samples.append(sample)

                        img_path = os.path.join(patient_root, filename)
                        out_dir = os.path.join(args.output_root, dirpath)
                        out_path = os.path.join(args.output_root,
                                                dirpath,
                                                filename)
                        os.makedirs(out_dir, exist_ok=True)
                        if args.enhance:
                            raise NotImplementedError
                        else:
                            os.symlink(os.path.realpath(img_path), out_path)
                    except ValueError:
                        print("Error processing '%s'" %
                              os.path.join(patient_root, filename))

        # Done processing patient. Can clear all the cached images.
        clear_image_cache()
    metadata = pd.DataFrame(samples).set_index("patient_id")
    metadata.to_csv(os.path.join(args.output_root, "metadata.csv"))
    print("Done.")


if __name__ == "__main__":
    main()
