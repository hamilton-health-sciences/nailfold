"""Explain input images by computing saliency maps."""

import argparse

import os

import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt

from extern.scorecam.utils import format_for_plotting, standardize_and_clip
from extern.scorecam.cam.scorecam import ScoreCAM

from nailfold import config
from nailfold.data import NailfoldDataModule
from nailfold.transforms import normalization_fn
from nailfold.utils import get_best_model, get_annotated_supervised_samples_for_patient


class LogisticToSoftmaxWrapper(nn.Module):
    """
    Converts the logistic (pre-sigmoid) outputs of a binary classification model
    to a multiclass classification-like output (pre-softmax) suitable for
    analysis in ScoreCAM.
    """

    def __init__(self, model, positive):
        super().__init__()

        self.model = model
        self.positive = positive

    def forward(self, x):
        pred = self.model(x)
        zero = torch.tensor([[0.]]).to(pred.device)
        if self.positive:
            logits = torch.cat((zero, pred), axis=1)
        else:
            logits = torch.cat((-pred, zero), axis=1)

        return logits

    @property
    def layer4(self):
        """
        Dummy pass-through layer to get the relevant layer of the underlying
        ResNet used for saliency map generation.
        """
        return self.model.wrapped.layer4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    parser.add_argument("--logs_root", type=str, default=config.RAY_LOGS_ROOT)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patient_id", type=int, default=None)
    parser.add_argument("--subset", type=str, choices=["train", "val", "test"],
                        default="val")
    parser.add_argument("--outcome_value", type=float, required=True)
    args = parser.parse_args()

    np.random.seed(args.repetition * config.NUM_FOLDS + args.fold)

    # Load the model
    model = get_best_model(args.logs_root,
                           args.outcome,
                           args.repetition,
                           args.fold)
    model.eval()

    # Wrap in ScoreCAM
    scorecam = ScoreCAM({
        "type": "resnet",
        "arch": LogisticToSoftmaxWrapper(model, bool(args.outcome_value)),
        "layer_name": "layer4",
        "input_size": config.INPUT_IMAGE_SIZE
    })

    # Get a random image
    dm = NailfoldDataModule(binary_covariate_name=args.outcome,
                            split_seed=args.repetition,
                            fold=args.fold,
                            data_root=args.data_root,
                            augmentation_fn=nn.Identity(),
                            normalization_fn=nn.Identity(),
                            num_workers=1)
    dm.setup()

    if args.subset == "train":
        subset = dm.train_set
    elif args.subset == "val":
        subset = dm.val_set
    elif args.subset == "test":
        subset = dm.test_set

    # If no patient ID is given, randomly select a patient ID from the subset
    if args.patient_id is None:
        subset_idx = subset.base.indices
        patient_ids = subset.base.dataset.patient_ids[subset_idx]
        outcomes = np.asarray([
            subset.base.dataset[idx][args.outcome]
            for idx in subset_idx
        ])
        patient_id_idxs = list(
            filter(lambda i: outcomes[i] == args.outcome_value,
                   range(len(outcomes)))
        )
        patient_id = patient_ids[np.random.choice(patient_id_idxs)]
        print("Selected patient ID: ", patient_id)
    else:
        patient_id = args.patient_id
    samples = get_annotated_supervised_samples_for_patient(dm, patient_id)

    # Put model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    predscores = []
    for sample in samples:
        outcome = (sample["outcome"] * 2) - 1.
        image = sample["image"]
        if torch.cuda.is_available():
            image = image.cuda()
        inp = normalization_fn(image).unsqueeze(0)
        pred = model(inp).detach().cpu().numpy()[0, 0]
        # We want the predscore to be higher for positive outcomes,
        # lower for negative outcomes
        predscore = pred * outcome
        predscores.append(predscore)

    sample_idx = np.random.choice(
        np.where(np.asarray(predscores) > 0.)[0]
    )
    sample = samples[sample_idx]

    print("Outcome: ", sample["outcome"])
    print("Image filepath: %s" % sample["image_filepath"])

    # Image & its transformed version for model input
    img = sample["image"]
    inp = normalization_fn(img).unsqueeze(0)

    # Put model & input on cuda
    if torch.cuda.is_available():
        model = model.cuda()
        inp = inp.cuda()

    pred = model(inp).detach().cpu().numpy()
    print("Prediction: ", pred)

    heatmap = scorecam(inp)

    if args.output_path is not None:
        results_root = args.output_path
    else:
        results_root = os.path.join(config.RESULTS_ROOT, args.outcome)
    os.makedirs(results_root, exist_ok=True)

    # Filenaming
    if args.outcome_value == 1.:
        designation = "case"
    else:
        designation = "control"

    # Plot - flip the channel axis so the color channels line up with
    # matplotlib expected order
    img_plot = format_for_plotting(torch.flip(img, dims=(0,)))
    heatmap_plot = format_for_plotting(standardize_and_clip(heatmap))

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.subplots_adjust(0,0,1,1)
    ax.axis("off")
    ax.imshow(img_plot)
    fig.savefig(
        os.path.join(
            results_root,
            "%s_patient_%d_raw.jpg" % (designation, patient_id)
        )
    )

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.subplots_adjust(0,0,1,1)
    ax.axis("off")
    ax.imshow(img_plot)
    ax.imshow(heatmap_plot, alpha=0.5)
    fig.savefig(
        os.path.join(
            results_root,
            "%s_patient_%d_saliency.jpg" % (designation, patient_id)
        )
    )


if __name__ == "__main__":
    main()
