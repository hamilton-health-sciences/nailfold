"""Basic computer vision backbones wrapped in a PyTorch Lightning interface."""

from typing import Callable, Optional, Dict, Any

from warnings import warn

import numpy as np

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torchvision import models

from .metric_helpers import build_binary_metrics, build_continuous_metrics


class TorchvisionWrapper(pl.LightningModule):
    """
    Wrap around a subset of the available models in TorchVision with a PyTorch
    Lightning interface to facilitate training, tuning and testing.
    """

    def __init__(self,
                 architecture: str,
                 pretrained: bool,
                 conditional_cov: bool,
                 optimizer: str,
                 learning_rate: float,
                 weight_decay: float,
                 exclude_biases_from_weight_decay: bool,
                 momentum: float,
                 nesterov: bool,
                 cosine_anneal_lr: bool,
                 head_dropout_prob: float,
                 head_hidden_dim: int,
                 case_weight: float,
                 eval_outcome_transform: Optional[Callable],
                 mode: str,
                 split_seed: int,
                 init_seed: int,
                 max_steps: int,
                 warmup_steps: int):
        """Initialize the wrapper.

        Args:
            architecture: The base architecture to use. See `torchvision.models`
                          for the list of available models.
            pretrained:  Whether or not to initialize the model using weights
                         pre-trained on ImageNet.
            optimizer: The optimizer to use, currently either "sgd" or "adam".
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
            exclude_biases_from_weight_decay: Whether or not to exclude bias
                                              parameters from weight decay.
            momentum: The momentum parameter for the optimizer, only used for
                      SGD.
            nesterov: Whether or not to use Nesterov momentum.
            cosine_anneal_lr: Whether or not to use cosine annealing of the
                              learning rate during training.
            case_weight: Cases will be weighted according to this value in the
                         loss.
            eval_outcome_transform: If not None, both the predicted outcome and
                                    the corresponding labels will be transformed
                                    using this function prior to feeding to the
                                    evaluation metrics. This is useful for e.g.
                                    normality transformations for continuous
                                    outcomes, but should not be necessary for
                                    binary outcomes. Note that the raw
                                    predictions and raw outcomes will still be
                                    used for the loss function.
            mode: Either "binary" or "continuous". If "binary", the predictions
                  will be fed to a loss function which sigmoid-transforms them
                  before computing the log/cross-entropy loss. If "continuous",
                  will be fed to the mean squared error loss. This parameter
                  also determines which metrics will be logged.
            split_seed: The seed use to split the data, accepted as an argument
                        only so that it's saved to generated hyperparameter
                        files.
            init_seed: The seed used to initialize the model.
        """
        super().__init__()

        self.mode = mode

        # Base parameters
        self.architecture = architecture
        self.pretrained = pretrained
        self.conditional_cov = conditional_cov

        # Optimizer & scheduler parameters
        if optimizer == "adam":
            assert momentum == 0., "Adam does not support a momentum parameter"
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.exclude_biases_from_weight_decay = exclude_biases_from_weight_decay
        self.cosine_anneal_lr = cosine_anneal_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        if not self.cosine_anneal_lr and self.warmup_steps > 0:
            warn("Cosine annealing was not enabled but arguments implied a "
                 "linear warmup. Linear warmup is only supported in "
                 "conjunction with cosine annealing, so this will be ignored.")

        # Loss weighting
        self.case_weight = case_weight

        # Outcome transform at evaluation time
        self.eval_outcome_transform = eval_outcome_transform

        # RNG seeds, not used in the code but accepted as arguments in order to
        # store them in the hyperparameter output files.
        self.split_seed = split_seed
        self.init_seed = init_seed

        # Save hyperparameters to an `hparams` file with the checkpoints for
        # easy access.
        self.save_hyperparameters()

        # Net
        self.wrapped: nn.Module = getattr(models, architecture)(
            pretrained=pretrained
        )
        if hasattr(self.wrapped, "fc"):
            print(
                f"'{architecture}' looks like a resnet. Replacing the fc layer."
            )
            original_head = self.wrapped.fc
            self.wrapped.fc = nn.Identity()
        elif hasattr(self.wrapped, "classifier"):
            print(
                f"'{architecture}' looks like a mobilenet. Replacing the fc "
                "layer."
            )
            original_head = self.wrapped.classifier[-1]
            self.wrapped.classifier[-1] = nn.Identity()
        else:
            raise ValueError("Not sure which is the final layer for "
                             f"architecture '{architecture}'.")
        if self.conditional_cov:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout_prob),
                nn.Linear(original_head.in_features + 1, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1)
            )
        else:
            self.head = nn.Linear(original_head.in_features, 1)

        # Metrics
        if self.mode == "binary":
            self.train_metrics = build_binary_metrics()
            self.validation_metrics = build_binary_metrics()
            self.test_metrics = build_binary_metrics()
        elif self.mode == "continuous":
            self.train_metrics = build_continuous_metrics()
            self.validation_metrics = build_continuous_metrics()
            self.test_metrics = build_continuous_metrics()

    def param_groups(self):
        """
        Extract the parameter groups during optimization.

        Used to exclude bias and batch normalization parameters from weight
        decay, which is a small tweak which helps training when weight decay
        is enabled.
        """
        if self.exclude_biases_from_weight_decay:
            if not hasattr(self.wrapped, "fc"):
                warn(
                    f"'{self.architecture}' does not look like a resnet. "
                    "Different models may name their bias and batch "
                    "normalization parameters differently, and the model "
                    "relies on these names to exclude these parameters from "
                    "weight decay. Thus weight decay exclusion may not be "
                    f"effective for '{self.architecture}'."
                )
            decay, no_decay = [], []
            for param_name, param in super().named_parameters():
                if ("bias" in param_name or
                    "bn" in param_name or
                    "downsample.1" in param_name):
                    no_decay.append(param)
                else:
                    decay.append(param)
            param_groups = [
                {"params": decay},
                {"params": no_decay, "weight_decay": 0.}
            ]
        else:
            param_groups = [
                {"params": super().parameters()}
            ]

        return param_groups

    def configure_optimizers(self):
        """
        Set up the optimizers and learning rate scheduler.
        """
        if self.optimizer == "adam":
            optimizer = Adam(self.param_groups(),
                             lr=self.learning_rate,
                             weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = SGD(self.param_groups(),
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay,
                            momentum=self.momentum,
                            nesterov=self.nesterov)
        optimizers = [optimizer]

        if self.cosine_anneal_lr:
            if self.warmup_steps == 0:
                schedulers = {
                    "scheduler": CosineAnnealingLR(
                        optimizer,
                        T_max=self.max_steps
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "strict": True
                }
            else:
                schedulers = {
                    "scheduler": LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=self.warmup_steps,
                        max_epochs=self.max_steps,
                        warmup_start_lr=1e-10
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "strict": True
                }
        else:
            schedulers = []

        return optimizers, schedulers

    def forward(self, x, conditional_cov=None):
        """
        Forward pass.

        Args:
            x: Input batch of images, with batch in the first dimension.
        Returns:
            preds: The corresponding predictions, with batch in the first
                   dimension.
        """
        features = self.wrapped(x)
        if self.conditional_cov and conditional_cov is not None:
            if len(conditional_cov.shape) == 1:
                if conditional_cov.shape[0] == features.shape[0]:
                    conditional_cov = conditional_cov.unsqueeze(1)
            preds = self.head(torch.cat((features, conditional_cov), axis=1))
        else:
            preds = self.head(features)

        return preds

    def shared_step(
        self,
        batch: Dict[str, Any],
        eval_outcome_transform: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Steps on the training, validation, and test sets.

        PyTorch Lightning automatically handles disabling gradients and
        BatchNorm updates in evaluation steps.

        Args:
            batch: The batch of input images, targets, patient IDs, and
                   conditional covariates (if using).
            eval_outcome_transform: Transform applied to predictions and targets
                                    prior to evaluation.

        Returns:
            ret: The loss, predictions, targets, patient IDs, and corresponding
                 filepaths of the input images.
        """
        sid = batch["id"]
        image_filepath = batch["image_filepath"]
        x = batch["image"]
        y = batch["outcome"]
        if self.conditional_cov:
            yhat = self.forward(x, batch["conditional_cov"])
        else:
            yhat = self.forward(x)

        if len(yhat.shape) > 1:
            yhat = yhat.squeeze(1)
        if len(y.shape) > 1:
            y = y.squeeze(1)

        ret = {}
        if self.mode == "binary":
            weight = torch.ones(len(y))
            weight[y.bool()] = self.case_weight

            loss_fn = nn.BCEWithLogitsLoss(weight=weight.to(yhat.device))
        else:
            loss_fn = nn.MSELoss()

        loss = loss_fn(yhat, y)

        # If provided, transform the outcomes so that downstream metrics will
        # be computed on the transformed outcome.
        if eval_outcome_transform is not None:
            y = eval_outcome_transform(y.detach().cpu())
            yhat = eval_outcome_transform(yhat.detach().cpu())

        ret["loss"] = loss
        ret["yhat"] = yhat.detach().cpu()
        ret["y"] = y.detach().cpu()
        ret["id"] = sid
        ret["image_filepath"] = image_filepath

        return ret

    # pylint: disable=unused-argument; batch_idx required in PL
    def training_step(self,
                      batch: Dict[str, Any],
                      batch_idx: int) -> Dict[str, Any]:
        """
        A single training step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            ret: The loss, predictions, and input batch elements needed for
                 evaluation.
        """
        return self.shared_step(batch)

    # pylint: disable=unused-argument; batch_idx required in PL
    def validation_step(self,
                        batch: Dict[str, Any],
                        batch_idx: int) -> Dict[str, Any]:
        """
        A single validation step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            ret: The loss, predictions, and input batch elements needed for
                 evaluation.
        """
        ret = self.shared_step(
            batch,
            eval_outcome_transform=self.eval_outcome_transform
        )

        return ret

    # pylint: disable=unused-argument; batch_idx required in PL
    def test_step(self,
                  batch: Dict[str, Any],
                  batch_idx: int) -> Dict[str, Any]:
        """
        A single test step.

        Args:
            batch: The input batch.

        Returns:
            ret: The loss, predictions, and input batch elements needed for
                 evaluation.
        """
        ret = self.shared_step(
            batch,
            eval_outcome_transform=self.eval_outcome_transform
        )

        return ret

    def training_step_end(self,
                          outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Training set metrics & logging callback.

        Logs the training loss and the training evaluation metrics.

        Args:
            outputs: The outputs of a step.

        Returns:
            outputs: Pass-through.
        """
        self.log("train_loss", outputs["loss"])
        self._update_metrics("train", self.train_metrics, outputs)

        return outputs

    def validation_step_end(self,
                            outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validation set metrics callback.

        Args:
            outputs: The outputs of a step.

        Returns:
            outputs: Pass-through.
        """
        self._update_metrics("validation", self.validation_metrics, outputs)

        return outputs

    def test_step_end(self,
                      outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test set metrics callback.

        Args:
            outputs: The outputs of a step.

        Returns:
            outputs: Pass-through.
        """
        self._update_metrics("test", self.test_metrics, outputs)

        return outputs

    def training_epoch_end(self,
                           outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Training set logging callback.

        Args:
            outputs: The outputs of all batches in an epoch.
        """
        self._log_metrics("train", self.train_metrics)

    def validation_epoch_end(self,
                             outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validation set metrics & logging callback.

        Args:
            outputs: The outputs of all batches in an epoch.
        """
        self._log_metrics("validation", self.validation_metrics)

        epoch_loss = torch.stack(
            [batch["loss"] for batch in outputs]
        ).mean()
        self.log("validation_loss", epoch_loss)

    def test_epoch_end(self,
                       outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test set logging callback.

        Args:
            outputs: The outputs of all batches in an epoch.
        """
        self._log_metrics("test", self.test_metrics)

        # Return outputs to enable downstream 95% CI computation
        outputs = {
            "id": torch.cat(list(map(lambda batch: batch["id"], outputs))),
            "image_filepath": np.concatenate(list(map(
                lambda batch: batch["image_filepath"], outputs
            ))),
            "yhat": torch.cat(list(map(lambda batch: batch["yhat"], outputs))),
            "y": torch.cat(list(map(lambda batch: batch["y"], outputs)))
        }

        return outputs

    def _update_metrics(self,
                        group: str,
                        metrics: nn.ModuleDict,
                        outputs: Dict[str, Any]) -> None:
        """
        Update a ModuleDict of metrics.

        Args:
            group: The group of metrics (either train, validation, or test).
            metrics: The dictionary mapping metric name to metric function.
            outputs: The outputs of a batch or epoch.
        """
        for metric_name, metric in metrics.items():
            if "img" in metric_name:
                metric.update(outputs["yhat"], outputs["y"])
            elif "patient" in metric_name:
                metric.update(outputs["id"], outputs["yhat"], outputs["y"])
            else:
                raise ValueError(
                    f"Not sure what to do with {group} metric {metric_name}."
                )

    def _log_metrics(self,
                     group: str,
                     metrics: nn.ModuleDict) -> None:
        """
        Log a ModuleDict of metrics.

        Args:
            group: The group of metrics (either train, validation, or test).
            metrics: The dictionary mapping metric name to metric function.
        """
        for metric_name, metric in metrics.items():
            try:
                if "roc" in metric_name and "auroc" not in metric_name:
                    fpr, tpr, _ = metric.compute()
                    metric_name = metric_name.replace("roc", "auroc")
                    metric = pl.metrics.functional.auc(fpr, tpr)
                self.log(f"{group}_{metric_name}", metric)
            except Exception:  # pylint: disable=broad-except
                pass
