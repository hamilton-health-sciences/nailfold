"""Used to aggregate a predictions to the patient level for evaluation."""

from typing import Optional, Callable, Any

import torch

from pytorch_lightning.metrics import Metric


class PatientAggregated(Metric):
    """
    Wraps a pre-existing metric and computes the metric on the patient-
    aggregated prediction.

    Used, for example, to first take the mean predicted probability across all
    input images for a given patient prior to computing the AUROC.
    """

    def __init__(self,
                 metric_fn: Callable,
                 pred_transform: Optional[Callable] = None,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None) -> None:
        """
        Args:
            metric: The underlying metric function.
            pred_transform: If given, will be called on both the prediction and
                            the target prior to computing the metric.
            dist_sync_on_step: See superclass.
            process_group: See superclass.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
        )

        self.metric_fn = metric_fn
        self.pred_transform = pred_transform

        self.add_state("ids", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self,
               ids: torch.Tensor,
               preds: torch.Tensor,
               target: torch.Tensor) -> None:
        """
        Args:
            ids: The patient IDs corresponding to each prediction/target pair.
            preds: The predictions.
            target: The targets.
        """
        self.ids.append(ids)
        if self.pred_transform:
            preds = self.pred_transform(preds)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        """
        Computes the metric.

        First, aggregates the mean predictions into the aggregated prediction
        for each patient. Then compares to the aggregated targets, which are
        just taken to be the first target given for a given patient (because
        targets should be the same for all patients).

        Returns:
            aggregated_metric: The aggregated metric. Can be NaN.
        """
        ids = torch.cat(self.ids, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        uniq_ids = torch.unique(ids)
        aggregated_preds = []
        aggregated_targets = []
        for uid in uniq_ids:
            sel = (ids == uid)
            aggregated_preds.append(torch.mean(preds[sel]))
            aggregated_targets.append(target[torch.nonzero(sel)[0]])
        aggregated_preds = torch.tensor(aggregated_preds)
        aggregated_targets = torch.tensor(aggregated_targets)

        try:
            metric = self.metric_fn(aggregated_preds,
                                    aggregated_targets)
        # pylint: disable=broad-except; better a NaN than a faulty metric or
        #                               needless error during training
        except Exception:
            metric = torch.tensor(float("nan"))

        return metric
