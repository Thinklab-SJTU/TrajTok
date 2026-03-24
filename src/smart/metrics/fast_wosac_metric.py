from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from google.protobuf import text_format
from torch import Tensor
from torchmetrics import Metric
from torch_geometric.utils import degree
from waymo_open_dataset.protos import sim_agents_metrics_pb2

from wosac_fast_eval_tool.fast_sim_agents_metrics import metrics as sim_agents_metric_api
from wosac_fast_eval_tool.scenario_gt_converter import gt_scenario_to_device


_METRIC_NAMES = [
    "metametric",
    "average_displacement_error",
    "min_average_displacement_error",
    "linear_speed_likelihood",
    "linear_acceleration_likelihood",
    "angular_speed_likelihood",
    "angular_acceleration_likelihood",
    "distance_to_nearest_object_likelihood",
    "collision_indication_likelihood",
    "time_to_collision_likelihood",
    "distance_to_road_edge_likelihood",
    "offroad_indication_likelihood",
    "simulated_collision_rate",
    "simulated_offroad_rate",
]

_2025_METRIC_NAMES = [
    "traffic_light_violation_likelihood",
    "simulated_traffic_light_violation_rate",
]


def _unwrap_gt_scenario(wrapper: Any) -> dict:
    if hasattr(wrapper, "value"):
        return wrapper.value
    return wrapper


class WOSACMetric(Metric):
    def __init__(
        self,
        version: str = "2025",
    ) -> None:
        super().__init__()
        self.version = version
        self.metric_names = _METRIC_NAMES
        if version == '2024':
            proto_path = 'wosac_fast_eval_tool/fast_sim_agents_metrics/challenge_2024_config.textproto'
        elif version == '2025':
            proto_path = 'wosac_fast_eval_tool/fast_sim_agents_metrics/challenge_2025_sim_agents_config.textproto'
            self.metric_names += _2025_METRIC_NAMES
        with open(proto_path,'r') as f:
            self.sim_agent_eval_config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), self.sim_agent_eval_config)

        for metric_name in self.metric_names:
            self.add_state(
                f"{metric_name}_sum",
                default=torch.tensor(0.0, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
        self.add_state(
            "count",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def _unbatch_agents(src: Tensor, batch: Tensor, dim: int) -> list[Tensor]:
        sizes = degree(batch, dtype=torch.long).tolist()
        return list(src.split(sizes, dim=dim))

    @torch.no_grad()
    def update(
        self,
        scenario_id: Iterable[str],
        gt_scenarios: Iterable[dict],
        agent_id: Tensor,
        agent_batch: Tensor,
        simulated_states: Tensor,
    ) -> None:

        device = simulated_states.device
        scenario_ids = list(scenario_id)
        gt_scenarios = list(gt_scenarios)
        agent_ids = self._unbatch_agents(agent_id, agent_batch, dim=0)
        sim_states = self._unbatch_agents(simulated_states, agent_batch, dim=1)

        for idx, _ in enumerate(scenario_ids):
            gt_scenario = _unwrap_gt_scenario(gt_scenarios[idx])
            gt_scenario = gt_scenario_to_device(gt_scenario, device=device)
            predict = {
                "agent_id": agent_ids[idx].to(device=device),
                "simulated_states": sim_states[idx].to(device=device),
            }
            scenario_metrics = sim_agents_metric_api.compute_scenario_metrics_for_bundle(
                self.sim_agent_eval_config,
                gt_scenario,
                predict,
                self.version,
            )
            for metric_name in self.metric_names:
                getattr(self, f"{metric_name}_sum").add_(
                    torch.tensor(
                        scenario_metrics[metric_name],
                        device=device,
                        dtype=torch.float64,
                    )
                )
            self.count += 1

    def compute(self) -> dict[str, float]:
        return {
            metric_name: (getattr(self, f"{metric_name}_sum") / self.count).item()
            for metric_name in self.metric_names
        }
