"""Infer hidden states (representations) for all images in the dataset."""

from argparse import ArgumentParser

from itertools import chain

import os

import pickle

from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from nailfold import config
from nailfold.models import FeatureExtractor
from nailfold.data import NailfoldDataModule
from nailfold.utils import get_best_model


def extract_features(torchvision_wrapper, dm):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    extractor = FeatureExtractor(torchvision_wrapper).to(device)
    extractor.eval()

    dm.setup()

    results = {
        "id": [],
        "image_path": [],
        "features": [],
        "split": ["train"] * len(dm.train_set) + ["val"] * len(dm.val_set) + \
                 ["test"] * len(dm.test_set)
    }
    all_data = chain(dm.train_dataloader(),
                     dm.val_dataloader(),
                     dm.test_dataloader())
    for batch in tqdm(all_data):
        image_paths = list(
            map(lambda f: os.path.join(*f.split("/")[-3:]),
                batch["image_filepath"])
        )
        results["id"].append(batch["id"])
        results["image_path"].append(image_paths)
        results["features"].append(
            extractor(batch["image"].to(device)).cpu().detach()
        )
    results["id"] = torch.cat(results["id"])
    results["image_path"] = np.concatenate(results["image_path"])
    results["features"] = torch.cat(results["features"], axis=0)

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--condition_on", type=str, default=None)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--logs_root", type=str, default="./ray_logs")
    args = parser.parse_args()

    if args.condition_on is not None:
        results_name = "%s_cond_%s" % (args.outcome, args.condition_on)
    else:
        results_name = args.outcome
    wrapper = get_best_model(args.logs_root,
                             results_name,
                             args.repetition,
                             args.fold)
    dm = NailfoldDataModule(binary_covariate_name=args.outcome,
                            condition_on=args.condition_on,
                            split_seed=args.repetition,
                            fold=args.fold,
                            augmentation_fn=nn.Identity())
    results = extract_features(wrapper, dm)

    output = {
        "outcome": args.outcome,
        "condition_on": args.condition_on,
        "repetition": args.repetition,
        "fold": args.fold,
        "num_folds": config.NUM_FOLDS,
        "results": results
    }

    results_root = os.path.join(config.RESULTS_ROOT, args.outcome)
    output_path = os.path.join(
        results_root,
        "representations_repetition=%d_fold=%d.pkl" % (args.repetition,
                                                       args.fold)
    )

    pickle.dump(output, open(output_path, "wb"))


if __name__ == "__main__":
    main()
