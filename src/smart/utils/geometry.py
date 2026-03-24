# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

import torch


def angle_between_2d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def clean_heading(valid: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    valid_pairs = valid[:, :-1] & valid[:, 1:]
    for i in range(heading.shape[1] - 1):
        heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
        change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
        heading[:, i + 1][change_needed] = heading[:, i][change_needed]
    return heading
