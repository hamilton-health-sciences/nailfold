"""Module for extracting hidden image features."""

from warnings import warn

import torch
from torch import nn

from .base import TorchvisionWrapper


class FeatureExtractor(nn.Module):
    """
    For a given `TorchvisionWrapper`, this module extracts the last hidden layer
    (image features) rather than the predictions. Useful for post-hoc analysis
    of internal states.
    """

    def __init__(self,
                 torchvision_wrapper: TorchvisionWrapper) -> None:
        """
        Args:
            torchvision_wrapper: The base model of interest.
        """
        super().__init__()

        self.features = torchvision_wrapper

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hidden features.

        Args:
            x: Batch of images with batch in the first dimension.

        Returns:
            representations: The image representations.
        """
        if self.training:
            warn("Warning: feature extractor is only intended to be used in "
                 "eval mode.")

        with torch.no_grad():
            representations = self.features.wrapped(x)

        return representations
