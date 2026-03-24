## Fast WOSAC Metric

This tool is an unofficial implementation of the WOSAC Metric evaluator, designed for rapid model testing and iteration.

### 使用方法：

The primary function for computing metrics is `wosac_fast_eval_tool.fast_sim_agents_metrics.compute_scenario_metrics_for_bundle`: 

```python

def compute_scenario_metrics_for_bundle(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig, ## WOSAC official configs
    gt_scenario: dict,       # obtain in preprocess with wosac_fast_eval_tool.scenario_gt_converter
    scenario_rollouts: dict, # 'simulated_states': Tensor [n_rollout, n_agents, n_step, (x,y,z,yaw)] i, 
                             # in global coordinate system； 'agent_id': Tensor [n_agents]
    version: str = '2025'    # support 2024 and 2025
) -> sim_agents_metrics_pb2.SimAgentMetrics:

```

**Online Evaluation Example** (default, evaluates immediately after each model inference step)： `src/samrt.metric/fast_wosac_metric.py` 
**Offline Evaluation Example** (saves all inference results first, then evaluates): `wosac_fast_eval_tool/fast_eval_offline.py`

### Features

* We rewrote the WOSAC metrics using PyTorch (originally implemented in TensorFlow) to ensure better compatibility with training frameworks and to facilitate GPU acceleration.

* We batch multiple rollouts of the same scenario together to maximize GPU utilization.

* We optimized the computational logic for map_metric (offroad_indication_likelihood, simulated_offroad_rate, traffic_light_violation_likelihood, simulated_traffic_light_violation_rate). Map polylines are segmented into equal lengths, and a hierarchical search is employed to accelerate the spatial query for the nearest polyline to the agents, significantly reducing VRAM consumption.

* We implemented seamless switching between the 2024 and 2025 versions of the WOSAC metrics.

### Speed

Official Implementation（CPU/GPU）： 14 s / scenario
Fast WOSAC Metric (1 x H100 GPU):  0.4 s / scenario (**35X Faster**)

### Accuracy

Across the entire validation set, the metrics computed using the official implementation versus our implementation on the same inference results are as follows:

| Metric Name | Official Implementation | Fast WOSAC Metric |  Absolute Error |
| :---: |:---: |:---: |:---: |
| metametric | 0.7833003739167209 | 0.7833004361195336| 6.220281267843575e-08 |
| average_displacement_error | 2.8609204922278995 | 2.8609205395894657| 4.736156622442422e-08 |
| min_average_displacement_error | 1.3413996061620501 | 1.3413996300401263| 2.3878076182981545e-08 |
| linear_speed_likelihood | 0.38566971245037057 | 0.3856697304710411 | 1.8020670511376125e-08 |
| linear_acceleration_likelihood | 0.3993123117931903 | 0.3993123920225635 | 8.022937320051327e-08 |
| angular_speed_likelihood | 0.5120824649694197 | 0.512082478557992 | 1.3588572378431252e-08 |
| angular_acceleration_likelihood |  0.6505992001960686 | 0.6505987461092263| 4.540868423497102e-07|
| distance_to_nearest_object_likelihood | 0.39117612289308307 | 0.3911761356506732| 1.275759015095801e-08 |
| collision_indication_likelihood | 0.9686309732356555| 0.9686310086013246| 3.5365669059927995e-08 |
| time_to_collision_likelihood | 0.8274178658994237| 0.8274179200481532| 5.414872950026961e-08 |
| distance_to_road_edge_likelihood | 0.682471707025851| 0.6824718727147412| 1.6568889016355115e-07 |
| offroad_indication_likelihood | 0.9547912612234546| 0.9547914733109856| 2.1208753098189703e-07 |
| traffic_light_violation_likelihood | 0.9815728407245071| 0.9815728914204771| 5.069596997753223e-08 |
| simulated_collision_rate | 0.04069170632494996| 0.04069170632494996| 0 |
| simulated_offroad_rate |  0.11372919733718174| 0.1137290201709346| 1.7716624714503304e-07 |
| simulated_traffic_light_violation_rate | 0.010714348735877283| 0.010714348735877283| 0  |

The error for all metrics is less than $10^{-6}$.

### Disclaimer

This is an unofficial tool intended for rapid performance evaluation and internal reference. For official competition submissions and final results, please rely exclusively on the official WOSAC evaluation server.