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

from typing import List

import torch
from torch import Tensor
from torch_geometric.utils import degree
from waymo_open_dataset.protos import sim_agents_submission_pb2


def _unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def get_scenario_rollouts(
    scenario_id: Tensor,  # [n_scenario, n_str_length]
    agent_id: Tensor,  # [n_agent]
    agent_batch: Tensor,  # [n_agent]
    pred_traj: Tensor,  # [n_agent, n_rollout, n_step, 2]
    pred_z: Tensor,  # [n_agent, n_rollout, n_step]
    pred_head: Tensor,  # [n_agent, n_rollout, n_step]
) -> List[sim_agents_submission_pb2.ScenarioRollouts]:
    scenario_id = scenario_id.cpu().numpy()
    agent_id = _unbatch(agent_id, agent_batch)
    pred_traj = _unbatch(pred_traj, agent_batch)
    pred_z = _unbatch(pred_z, agent_batch)
    pred_head = _unbatch(pred_head, agent_batch)
    agent_id = [x.cpu().numpy() for x in agent_id]
    pred_traj = [x.cpu().numpy() for x in pred_traj]
    pred_z = [x.cpu().numpy() for x in pred_z]
    pred_head = [x.cpu().numpy() for x in pred_head]

    n_scenario = scenario_id.shape[0]
    n_rollout = pred_traj[0].shape[1]
    scenario_rollouts = []
    for i_scenario in range(n_scenario):
        joint_scenes = []
        for i_rollout in range(n_rollout):
            simulated_trajectories = []
            for i_agent in range(len(agent_id[i_scenario])):
                simulated_trajectories.append(
                    sim_agents_submission_pb2.SimulatedTrajectory(
                        center_x=pred_traj[i_scenario][i_agent, i_rollout, :, 0],
                        center_y=pred_traj[i_scenario][i_agent, i_rollout, :, 1],
                        center_z=pred_z[i_scenario][i_agent, i_rollout],
                        heading=pred_head[i_scenario][i_agent, i_rollout],
                        object_id=agent_id[i_scenario][i_agent],
                    )
                )
            joint_scenes.append(
                sim_agents_submission_pb2.JointScene(
                    simulated_trajectories=simulated_trajectories
                )
            )

        _str_scenario_id = "".join([chr(x) for x in scenario_id[i_scenario] if x > 0])
        scenario_rollouts.append(
            sim_agents_submission_pb2.ScenarioRollouts(
                joint_scenes=joint_scenes, scenario_id=_str_scenario_id
            )
        )

    return scenario_rollouts


def get_scenario_id_int_tensor(scenario_id: List[str], device: torch.device) -> Tensor:
    scenario_id_int_tensor = []
    for str_id in scenario_id:
        int_id = [-1] * 16  # max_len of scenario_id string is 16
        for i, c in enumerate(str_id):
            int_id[i] = ord(c)
        scenario_id_int_tensor.append(
            torch.tensor(int_id, dtype=torch.int32, device=device)
        )
    return torch.stack(scenario_id_int_tensor, dim=0)  # [n_scenario, 16]
