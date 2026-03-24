from typing import List

from google.protobuf import text_format
import numpy as np
import torch
import time
# copybara removed file resource import
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from . import estimators
from . import metric_features
from . import trajectory_features

_METRIC_FIELD_NAMES_BY_BUCKET = {
    'kinematic': [
            'linear_speed', 'linear_acceleration',
            'angular_speed', 'angular_acceleration',
    ],
    'interactive': [
            'distance_to_nearest_object', 'collision_indication',
            'time_to_collision',
    ],
    'map_based': [
            'distance_to_road_edge', 'offroad_indication',
            'traffic_light_violation',
    ]
}
_METRIC_FIELD_NAMES = (
    _METRIC_FIELD_NAMES_BY_BUCKET['kinematic'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['interactive'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['map_based']
)

@torch.no_grad()
def compute_scenario_metrics_for_bundle(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig, ## WOSAC official
    gt_scenario: dict, # obtain in preprocess
    scenario_rollouts: dict, # 'simulated_states': Tensor [n_rollout, n_agents, n_step, (x,y,z,yaw)], 
                             # 'agent_id':Tensor [n_agents]
    version: str = '2025'    # support 2024 and 2025
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    """Computes the scenario-level metrics for the given bundle."""
    # Computes the metric features for log and sim.
    log_features, sim_features, valid_masks = (
        metric_features.compute_scenario_rollouts_features(
            gt_scenario, scenario_rollouts, version)
        )
    # ==== Average Displacement Error ====
    # This metric is not included in the scoring meta-metric, but we report it
    # to have a baseline comparison with existing Behaviour Prediction challenges.
    # We report both ADE (averaged over simulations and objects) and minADE
    # (averaged over objects, minimum over simulations).
    average_displacement_error = torch.mean(
        sim_features['average_displacement_error'])
    min_average_displacement_error = torch.min(
        torch.mean(sim_features['average_displacement_error'], dim=1))

    # ==== Dynamics features ====
    # Compute the log-likelihoods of speed features (first derivatives).
    # Note: For log_values we take only index=0 of the batch dimension to have
    # shape (n_objects, n_steps), as specified by
    # `log_likelihood_estimate_timeseries()`.
    linear_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
        feature_config=config.linear_speed,
        log_values=log_features['linear_speed'][0],
        sim_values=sim_features['linear_speed'])
    angular_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
        feature_config=config.angular_speed,
        log_values=log_features['angular_speed'][0],
        sim_values=sim_features['angular_speed'])
    # Get the log speed (linear and angular) validity. Since this is computed by
    # a delta between steps `i-1` and `i+1`, we verify that both of these are
    # valid (logical and).
    # The score is computed as the sum of the log-likelihoods, filtered by
    # validity. We exponentiate the result to get a score in the range [0,1].
    linear_speed_likelihood = torch.exp(_reduce_average_with_validity(
        linear_speed_log_likelihood, log_features['speed_validity']))
    angular_speed_likelihood = torch.exp(_reduce_average_with_validity(
        angular_speed_log_likelihood, log_features['speed_validity']))

    # Similarly, we compute likelihoods for acceleration features. This time,
    # we have to compute a double-step validity mask, because accelerations
    # involve the validity of `i`, `i+1`, `i+2`.
    linear_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
        feature_config=config.linear_acceleration,
        log_values=log_features['linear_acceleration'][0],
        sim_values=sim_features['linear_acceleration'])
    angular_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
        feature_config=config.angular_acceleration,
        log_values=log_features['angular_acceleration'][0],
        sim_values=sim_features['angular_acceleration'])
    linear_accel_likelihood = torch.exp(_reduce_average_with_validity(
        linear_accel_log_likelihood, log_features['acceleration_validity']))
    angular_accel_likelihood = torch.exp(_reduce_average_with_validity(
        angular_accel_log_likelihood, log_features['acceleration_validity']))

    # Collision likelihood is computed by aggregating in time. For invalid objects
    # in the logged scenario, we need to filter possible collisions in simulation.
    # `sim_collision_indication` shape: (n_samples, n_objects).
    sim_collision_indication = torch.any(
        torch.where(valid_masks, sim_features['collision_per_step'], 
        torch.zeros_like(sim_features['collision_per_step'], dtype=torch.bool)), dim=2)
    log_collision_indication = torch.any(
        torch.where(valid_masks, log_features['collision_per_step'], 
        torch.zeros_like(log_features['collision_per_step'], dtype=torch.bool)), dim=2)
    # Collision and distance to other objects. Again, aggregate over objects and
    # timesteps by summing the log-probabilities.
    collision_score = estimators.log_likelihood_estimate_scenario_level(
        feature_config=config.collision_indication,
        log_values=log_collision_indication[0],
        sim_values=sim_collision_indication
    )
    collision_likelihood = torch.exp(torch.mean(collision_score))
    distance_to_objects_log_likelihood = (
        estimators.log_likelihood_estimate_timeseries(
            feature_config=config.distance_to_nearest_object,
            log_values=log_features['distance_to_nearest_object'][0],
            sim_values=sim_features['distance_to_nearest_object'],
        )
    )
    distance_to_obj_likelihood = torch.exp(
        _reduce_average_with_validity(
            distance_to_objects_log_likelihood, valid_masks
        )
    )
    ttc_log_likelihood = estimators.log_likelihood_estimate_timeseries(
        feature_config=config.time_to_collision,
        log_values=log_features['time_to_collision'][0],
        sim_values=sim_features['time_to_collision'],
    )
    if version == '2025': # for 2025 version, only compute TTC for vehicles
        is_vehicle = torch.eq(log_features['object_type'], scenario_pb2.Track.ObjectType.TYPE_VEHICLE)
        ttc_validity = torch.logical_and(valid_masks, is_vehicle.unsqueeze(1))
    else:
        ttc_validity = valid_masks

    ttc_likelihood = torch.exp(_reduce_average_with_validity(ttc_log_likelihood, ttc_validity))
    # Off-road and distance to road edge. Again, aggregate over objects and
    # timesteps by summing the log-probabilities.
    # `sim_offroad_indication` shape: (n_samples, n_objects).
    sim_offroad_indication = torch.any(
        torch.where(valid_masks, sim_features['offroad_per_step'], 
        torch.zeros_like(sim_features['offroad_per_step'], dtype=torch.bool)),dim=2)
    log_offroad_indication = torch.any(
        torch.where(valid_masks, log_features['offroad_per_step'], 
        torch.zeros_like(log_features['offroad_per_step'], dtype=torch.bool)),dim=2)
    offroad_score = estimators.log_likelihood_estimate_scenario_level(
        feature_config=config.offroad_indication,
        log_values=log_offroad_indication[0],
        sim_values=sim_offroad_indication,
    )
    offroad_likelihood = torch.exp(torch.mean(offroad_score))

    # `distance_to_road_edge_log_likelihood` shape: (n_objects, n_steps).
    distance_to_road_edge_log_likelihood = (
        estimators.log_likelihood_estimate_timeseries(
            feature_config=config.distance_to_road_edge,
            log_values=log_features['distance_to_road_edge'][0],
            sim_values=sim_features['distance_to_road_edge'],
        )
    )
    distance_to_road_edge_likelihood = torch.exp(
        _reduce_average_with_validity(distance_to_road_edge_log_likelihood, valid_masks)
    )
    # Traffic light violation. This is treated like collision and offroad,
    # aggregated over time for any violation for each object (filtered by
    # validity) and scored against log. Filter for vehicles only.
    # Shape: (n_rollout,n_agent,n_step).
    
    if version == '2025':
        device = log_features['traffic_light_violation_per_step'].device
        tl_validity = tl_validity = torch.logical_and(
            valid_masks,
            is_vehicle.unsqueeze(1)
            ).to(device) #[n_agent,n_step]
        log_traffic_light_violation_indication = torch.any(
            torch.logical_and(
                log_features['traffic_light_violation_per_step'], #[1,n_agent,n_step]
                tl_validity.unsqueeze(0)), dim=2)
        #[1,n_agent]
        sim_traffic_light_violation_indication = torch.any(
            torch.logical_and(
                sim_features['traffic_light_violation_per_step'], 
                tl_validity.unsqueeze(0)), dim=2)
        #[n_rollout,n_agent]
        tl_score = estimators.log_likelihood_estimate_scenario_level(
            feature_config=config.traffic_light_violation,
            log_values=log_traffic_light_violation_indication[0],
            sim_values=sim_traffic_light_violation_indication,
            )
        #   print(gt_scenario['scenario_id'],":",tl_score)
        tl_likelihood = torch.exp(torch.mean(tl_score))
    # ==== Simulated collision and offroad rates ====
    simulated_collision_rate = torch.sum(
            sim_collision_indication.to(torch.int32)
    ) / torch.sum(torch.ones_like(sim_collision_indication, dtype=torch.int32))
    simulated_offroad_rate = torch.sum(
            sim_offroad_indication.to(torch.int32)
    ) / torch.sum(torch.ones_like(sim_offroad_indication, dtype=torch.int32))
    device = sim_features['traffic_light_violation_per_step'].device
    mask = valid_masks.to(device)
    traffic_light_violation_indication = torch.any(
        sim_features['traffic_light_violation_per_step'] & mask.unsqueeze(0), dim=2)
    traffic_light_violation_rate = torch.sum(
        traffic_light_violation_indication.int()
        ) / torch.sum(
        torch.ones_like(traffic_light_violation_indication, dtype=torch.int32)
        )
    # ==== Meta-metric ====

    likelihood_metrics = {
        'linear_speed_likelihood': linear_speed_likelihood.item(),
        'linear_acceleration_likelihood': linear_accel_likelihood.item(),
        'angular_speed_likelihood': angular_speed_likelihood.item(),
        'angular_acceleration_likelihood': angular_accel_likelihood.item(),
        'distance_to_nearest_object_likelihood': distance_to_obj_likelihood.item(),
        'collision_indication_likelihood': collision_likelihood.item(),
        'time_to_collision_likelihood': ttc_likelihood.item(),
        'distance_to_road_edge_likelihood': distance_to_road_edge_likelihood.item(),
        'offroad_indication_likelihood': offroad_likelihood.item(),
    }
    
    if version == '2025':
        likelihood_metrics['traffic_light_violation_likelihood'] = tl_likelihood.item()

    metametric = _compute_metametric(
            config, sim_agents_metrics_pb2.SimAgentMetrics(**likelihood_metrics))
    likelihood_metrics.update({
            'metametric': metametric,
            'average_displacement_error': average_displacement_error.item(),
            'min_average_displacement_error': min_average_displacement_error.item(),
            'simulated_collision_rate': simulated_collision_rate.item(),
            'simulated_offroad_rate': simulated_offroad_rate.item(),
            'simulated_traffic_light_violation_rate': traffic_light_violation_rate.item(),
    })
    
    return likelihood_metrics

def aggregate_scenario_metrics(
        all_scenario_metrics: List[sim_agents_metrics_pb2.SimAgentMetrics]
        ) -> sim_agents_metrics_pb2.SimAgentMetrics:
    """Aggregates the per-scenario metrics over the whole dataset."""
    msg_fields = [field[0].name for field in all_scenario_metrics[0].ListFields()]
    field_values = {field_name: [] for field_name in msg_fields}
    for scenario_metrics in all_scenario_metrics:
        for field_name in msg_fields:
            field_values[field_name].append(getattr(scenario_metrics, field_name))
    # Remove the scenario ID field.
    del field_values['scenario_id']
    # Average all the fields.
    field_values = {
            name: np.mean(values) for (name, values) in field_values.items()}
    return sim_agents_metrics_pb2.SimAgentMetrics(
            **field_values)

def aggregate_metrics_to_buckets(
        config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
        metrics: sim_agents_metrics_pb2.SimAgentMetrics
) -> sim_agents_metrics_pb2.SimAgentsBucketedMetrics:
    """Aggregates metrics into buckets for better readability."""
    bucketed_metrics = {}
    for bucket_name, fields_in_bucket in _METRIC_FIELD_NAMES_BY_BUCKET.items():
        weighted_metric, weights_sum = 0.0, 0.0
        for field_name in fields_in_bucket:
            likelihood_field_name = field_name + '_likelihood'
            weight = getattr(config, field_name).metametric_weight
            metric_score = getattr(metrics, likelihood_field_name)
            weighted_metric += weight * metric_score
            weights_sum += weight
        if weights_sum == 0:
            raise ValueError('The bucket\'s weight sum is zero. Check your metrics'
                                             ' config.')
        bucketed_metrics[bucket_name] = weighted_metric / weights_sum

    return sim_agents_metrics_pb2.SimAgentsBucketedMetrics(
            realism_meta_metric=metrics.metametric,
            kinematic_metrics=bucketed_metrics['kinematic'],
            interactive_metrics=bucketed_metrics['interactive'],
            map_based_metrics=bucketed_metrics['map_based'],
            min_ade=metrics.min_average_displacement_error,
            simulated_collision_rate=metrics.simulated_collision_rate,
            simulated_offroad_rate=metrics.simulated_offroad_rate,
            simulated_traffic_light_violation_rate=(
                    metrics.simulated_traffic_light_violation_rate),
    )

def _reduce_average_with_validity(
        tensor: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    """Returns the tensor's average, only selecting valid items.

    Args:
        tensor: A float tensor of any shape.
        validity: A boolean tensor of the same shape as `tensor`.

    Returns:
        A float tensor of shape (1,), containing the average of the valid elements
        of `tensor`.
    """
    if tensor.shape != validity.shape:
        raise ValueError('Shapes of `tensor` and `validity` must be the same.'
                                         f'(Actual: {tensor.shape}, {validity.shape}).')
    cond_sum = torch.sum(torch.where(validity, tensor, torch.zeros_like(tensor)))
    valid_sum = torch.sum(validity.to(torch.float32))
    return cond_sum / valid_sum

def _compute_metametric(
        config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
        metrics: sim_agents_metrics_pb2.SimAgentMetrics,
):
    """Computes the meta-metric aggregation."""
    metametric = 0.0
    for field_name in _METRIC_FIELD_NAMES:
        likelihood_field_name = field_name + '_likelihood'
        weight = getattr(config, field_name).metametric_weight
        metric_score = getattr(metrics, likelihood_field_name)
        metametric += weight * metric_score
    return metametric
