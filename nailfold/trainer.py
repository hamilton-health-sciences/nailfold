"""A small modification to the PL trainer class."""

import torch

from pytorch_lightning import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """
    Override the builtin Trainer class to support returning tensors in the
    final test results in order to bootstrap confidence intervals around
    mean cross-validated performance metrics.
    """

    def run_test(self):
        # only load test dataloader for testing
        # self.reset_test_dataloader(ref_model)
        with self.profiler.profile("run_test_evaluation"):
            eval_loop_results, _ = self.run_evaluation()

        if len(eval_loop_results) == 0:
            return 1

        # remove the tensors from the eval results
        for result in eval_loop_results:
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            result[k] = v.cpu().item()
                        else:
                            result[k] = v.cpu()

        return eval_loop_results
