from typing import Tuple

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle


@torch.no_grad()
def get_prob_targets(
    gt_idx: Tensor,  # [n_agent, n_step]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
) -> Tensor:  # [n_agent, n_step, n_token] prob, last dim sum up to 1

    closest_token_mask = one_hot(gt_idx, num_classes=token_traj.shape[1]).to(bool)
    prob_target = torch.zeros(gt_idx.shape[0], gt_idx.shape[1], token_traj.shape[1], device=gt_idx.device)
    prob_target[closest_token_mask] = 1
    return prob_target


@torch.no_grad()
def get_prob_targets_spatial_aware_smoothing(
    gt_idx: Tensor,  # [n_agent, n_step]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    label_smoothing: float
) -> Tensor:  # [n_agent, n_step, n_token] prob, last dim sum up to 1
    # ! tokenize to index, then compute prob

    gt_token_traj = torch.gather(token_traj, dim=1, index=gt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 2))
    dists = torch.norm(gt_token_traj[:, :, None, :, :] - token_traj[:, None, :, :, :], dim=-1).mean(-1)
    closest_token_mask = one_hot(gt_idx, num_classes=token_traj.shape[1]).to(bool)
    prob_target = torch.zeros(gt_idx.shape[0], gt_idx.shape[1], token_traj.shape[1], device=gt_idx.device)
    prob_target[closest_token_mask] = 1 - label_smoothing
    proj = 1 / ((0.0001 + dists) ** 2)
    proj = proj * (~closest_token_mask).int()
    proj_sum = proj.sum(dim=-1, keepdim=True) 
    proj = proj / proj_sum 
    prob_target += proj / proj_sum * label_smoothing

    return prob_target


