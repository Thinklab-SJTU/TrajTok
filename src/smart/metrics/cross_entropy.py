from typing import Optional

import torch
from torch import Tensor, tensor
from torch.nn.functional import cross_entropy
from torchmetrics.metric import Metric

from .utils import get_prob_targets, get_prob_targets_spatial_aware_smoothing
from src.smart.utils.split_and_merge import split_by_type, merge_by_type

class CrossEntropy(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        label_smoothing: float,
        spatial_aware_smoothing: bool = False,
    ) -> None:
        super().__init__()

        self.label_smoothing = label_smoothing
        self.spatial_aware_smoothing = spatial_aware_smoothing
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")
 
    def update(
        self,
        # ! action that goes from [(10->15), ..., (85->90)]
        next_token_logits: dict[str, Tensor],  # [n_agent, 16, n_token]
        next_token_valid: Tensor,  # [n_agent, 16]
        gt_idx: Tensor, # [n_agent, 16]
        gt_valid_mask: Tensor, # [n_agent, 16]
        token_agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
        # ! for filtering intersting agent for training
        train_mask: Optional[Tensor] = None,  # [n_agent]
        # ! for rollout_as_gt
        type_mask = None,
        **kwargs,
    ) -> None:

        gt_idx_by_type = split_by_type(gt_idx, type_mask)
        loss_by_type = {}
        for agent_type, mask in type_mask.items():
            if mask.sum() == 0:
                continue
            if self.spatial_aware_smoothing:
                prob_target_by_type = get_prob_targets_spatial_aware_smoothing(
                    gt_idx=gt_idx_by_type[agent_type],
                    token_traj=token_traj[agent_type],
                    label_smoothing=self.label_smoothing,
                )
            else:
                prob_target_by_type = get_prob_targets(
                    gt_idx=gt_idx_by_type[agent_type],
                    token_traj=token_traj[agent_type],
                )
            loss_by_type[agent_type] = cross_entropy(
                next_token_logits[agent_type].transpose(1, 2),  # [n_agent, n_token, n_step], logits
                prob_target_by_type.transpose(1, 2),  # [n_agent, n_token, n_step], prob
                reduction="none",
                label_smoothing=self.label_smoothing if not self.spatial_aware_smoothing else 0,
            )

        loss = merge_by_type(loss_by_type, type_mask)
        # ! weighting final loss [n_agent, n_step]
        loss_weighting_mask = next_token_valid & gt_valid_mask
        if self.training:
            loss_weighting_mask &= train_mask.unsqueeze(1)  # [n_agent, n_step]

        self.loss_sum += (loss * loss_weighting_mask).sum()
        self.count += (loss_weighting_mask > 0).sum()

    def compute(self) -> Tensor:
        return self.loss_sum / self.count
    
    

