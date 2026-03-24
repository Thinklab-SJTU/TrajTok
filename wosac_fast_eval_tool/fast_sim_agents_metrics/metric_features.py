from __future__ import annotations

import collections
import dataclasses
import time
import torch
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from . import interaction_features
from . import map_metric_features
from . import trajectory_features
from . import traffic_light_features
_ChallengeType = submission_specs.ChallengeType
_LaneType = map_pb2.LaneCenter.LaneType
import time
_distance_computation_total_time = 0.0
_distance_computation_call_count = 0
@dataclasses.dataclass(frozen=True)
class MetricFeatures:
    """Collection of features used to compute sim-agent metrics.

    These features may be a function of simulated data (e.g. dynamics and
    collisions), logged data (e.g. displacement) and map features (e.g. offroad).

    This class can be used to represent both features coming from the original
    Scenario and features from simulation. The samples dimension is set
    accordingly depending on the source (n_samples=1 for log and n_samples=32 for
    simulation).

    Some of the features are computed in 3D (x/y/z) to have better consistency
    with the original data and making these metrics more suitable for future
    updates.

    Attributes:
        object_id: A tensor of shape (n_objects,), containing the integer IDs of all
            the evaluated objects. The object_id tensor is not batched because all the
            objects need to be consistent over samples for proper evaluation.
        valid: Boolean tensor of shape (n_samples, n_objects, n_steps), identifying
            which objects are valid over time. This is used to filter the features
            when computing metrics.
        average_displacement_error: Per-object average (over time) displacement
            error compared to the logged trajectory. Shape: (n_samples, n_objects).
        linear_speed: Linear speed in 3D computed as the 1-step difference between
            trajectory points. Shape: (n_samples, n_objects, n_steps).
        linear_acceleration: Linear acceleration in 3D computed as the 1-step
            difference between speeds of objects.
            Shape: (n_samples, n_objects, n_steps).
        angular_speed: Angular speed computed as the 1-step difference in heading.
            Shape: (n_samples, n_objects, n_steps).
        angular_acceleration: Angular acceleration computed as the 1-step difference
            in angular_speed. Shape: (n_samples, n_objects, n_steps).
        distance_to_nearest_object: Signed distance (in meters) to the nearest
            object in the scene. Shape: (n_samples, n_objects, n_steps).
        collision_per_step: Boolean tensor indicating whether the object collided,
            with any other object. Shape: (n_samples, n_objects, n_steps).
        time_to_collision: Time (in seconds) before the object collides with the
            object it is following (if it exists), assuming constant speeds.
            Shape: (n_samples, n_objects, n_steps).
        distance_to_road_edge: Signed distance (in meters) to the nearest road edge
            in the scene. Shape: (n_samples, n_objects, n_steps).
        offroad_per_step: Boolean tensor indicating whether the object went
            off-road. Shape: (n_samples, n_objects, n_steps).
    """
    object_id: torch.Tensor
    object_type: torch.Tensor
    valid: torch.Tensor
    average_displacement_error: torch.Tensor
    linear_speed: torch.Tensor
    linear_acceleration: torch.Tensor
    angular_speed: torch.Tensor
    angular_acceleration: torch.Tensor
    distance_to_nearest_object: torch.Tensor
    collision_per_step: torch.Tensor
    time_to_collision: torch.Tensor
    distance_to_road_edge: torch.Tensor
    offroad_per_step: torch.Tensor
    traffic_light_violation_per_step: torch.Tensor


def compute_metric_features(
        object_ids,
        object_types,
        simulated_all_trajectories,
        simulated_val_trajectories,
        logged_val_trajectories,
        logged_val_trajectorie_masks,
        logged_all_trajectories,
        logged_all_trajectorie_masks,
        evaluated_object_mask,
        road_edges,
        use_log_validity,
        lane_ids, 
        lane_polylines, 
        traffic_signals,
        version
) -> dict:
    
    if simulated_all_trajectories.shape[-1] == 9:
        simulated_all_trajectories = simulated_all_trajectories[...,[0,1,2,6]]
        simulated_val_trajectories = simulated_val_trajectories[...,[0,1,2,6]]
    if simulated_all_trajectories.shape[-2] == 91:
        simulated_all_trajectories = simulated_all_trajectories[:,:,11:,:]
        simulated_val_trajectories = simulated_val_trajectories[:,:,11:,:]

    logged_all_trajectories_future = logged_all_trajectories[:,11:,:]
    logged_all_trajectories_future_masks = logged_all_trajectorie_masks[:,11:]
    logged_val_trajectories_future_masks = logged_val_trajectorie_masks[:,11:]

    if use_log_validity:
        valid_mask = logged_all_trajectories_future_masks
    else:
        valid_mask = torch.ones_like(logged_all_trajectories_future_masks).bool()
    
    simulated_all_trajectories_with_gt_history = torch.cat([logged_all_trajectories[:,:11,[0,1,2,6]].unsqueeze(0).repeat(simulated_all_trajectories.shape[0],1,1,1), simulated_all_trajectories],dim=-2)
    simulated_val_trajectories_with_gt_history = torch.cat([logged_val_trajectories[:,:11,[0,1,2,6]].unsqueeze(0).repeat(simulated_val_trajectories.shape[0],1,1,1), simulated_val_trajectories],dim=-2)

    displacement_error = torch.norm(simulated_val_trajectories_with_gt_history[...,0:3] - logged_val_trajectories[torch.newaxis,:,:,0:3], dim=-1)
    ade_masks = logged_val_trajectorie_masks.unsqueeze(0).repeat(simulated_val_trajectories.shape[0],1,1)
    ades = torch.sum(torch.where(ade_masks, displacement_error, 0), dim=-1) / ade_masks.sum(dim=-1)
    

    # Kinematics-related features, i.e. speed and acceleration, both linear and
    # angular. These feature are computed as finite differences of the objects
    # position, which makes the first step invalid. We prepend the history steps
    # so that this first simulation step has a valid difference too.
    linear_speed, linear_accel, angular_speed, angular_accel = (
        trajectory_features.compute_kinematic_features(
            simulated_val_trajectories_with_gt_history,
            seconds_per_step=0.1))
    linear_speed = linear_speed[:,:,11:]
    linear_accel = linear_accel[:,:,11:]
    angular_speed = angular_speed[:,:,11:]
    angular_accel = angular_accel[:,:,11:]
    speed_validity, acceleration_validity = trajectory_features.compute_kinematic_validity(logged_val_trajectories_future_masks)

    # Interactive features are computed between all simulated objects, but only
    # scored for evaluated objects.
    distances_to_objects = (
        interaction_features.compute_distance_to_nearest_object(
            boxes=torch.cat([simulated_all_trajectories[...,0:3], 
                             logged_all_trajectories_future[...,3:6].squeeze(0).repeat(simulated_all_trajectories.shape[0],1,1,1), 
                             simulated_all_trajectories[...,[3]]],dim=-1),
            valid=valid_mask,
            evaluated_object_mask=evaluated_object_mask
            ))
    is_colliding_per_step = torch.less(
        distances_to_objects, interaction_features.COLLISION_DISTANCE_THRESHOLD)

    times_to_collision = (
        interaction_features.compute_time_to_collision_with_object_in_front(
            center_x=simulated_all_trajectories_with_gt_history[...,0],
            center_y=simulated_all_trajectories_with_gt_history[...,1],
            length=logged_all_trajectories_future[...,3].squeeze(0).repeat(simulated_all_trajectories.shape[0],1,1),
            width=logged_all_trajectories_future[...,4].squeeze(0).repeat(simulated_all_trajectories.shape[0],1,1),
            heading=simulated_all_trajectories_with_gt_history[...,3],
            valid=valid_mask,
            evaluated_object_mask=evaluated_object_mask,
            seconds_per_step=0.1,
        )
    )

    distances_to_road_edge = map_metric_features.compute_distance_to_road_edge(
        boxes=torch.cat([simulated_all_trajectories[...,0:3], logged_all_trajectories_future[...,3:6].squeeze(0).repeat(simulated_all_trajectories.shape[0],1,1,1), simulated_all_trajectories[...,[3]]],dim=-1),
        valid=valid_mask,
        evaluated_object_mask=evaluated_object_mask,
        road_edge_polylines=road_edges,
    )
 
    if version == '2025' and lane_polylines and traffic_signals:
        if use_log_validity:
            traffic_light_valid = logged_all_trajectorie_masks
        else:
            traffic_light_valid = torch.cat(
                [
                    logged_all_trajectorie_masks[:, :11],
                    torch.ones_like(logged_all_trajectorie_masks[:, 11:], dtype=torch.bool),
                ],
                dim=1,
            )
        red_light_violations = traffic_light_features.compute_red_light_violation(
            center_x=simulated_all_trajectories_with_gt_history[...,0],
            center_y=simulated_all_trajectories_with_gt_history[...,1],
            valid=traffic_light_valid,
            evaluated_object_mask=evaluated_object_mask,
            lane_polylines=lane_polylines,
            lane_ids=lane_ids,
            traffic_signals=traffic_signals
        )[:, :, 11:] #[n_rollout,n_agent,n_step]
    else:
        # Cannot compute red light violations without lanes and traffic signals,
        # so we assume no violations.
        evaluated_object_indices = torch.where(evaluated_object_mask)[0]
        red_light_violations = torch.zeros(
            (
                simulated_all_trajectories.shape[0],
                len(evaluated_object_indices),
                simulated_all_trajectories.shape[2],
            ),
            dtype=torch.bool,
            device=simulated_all_trajectories.device,
        )

    is_offroad_per_step = torch.greater(
        distances_to_road_edge, map_metric_features.OFFROAD_DISTANCE_THRESHOLD
    )
 
    # Pack into `MetricFeatures`, also adding a batch dimension of 1 (except for
    # `object_id`).
    return {
            'object_id':object_ids,
            'object_type':object_types,
            'valid':valid_mask,
            'average_displacement_error':ades,
            'linear_speed':linear_speed,
            'linear_acceleration':linear_accel,
            'angular_speed':angular_speed,
            'angular_acceleration':angular_accel,
            'distance_to_nearest_object':distances_to_objects,
            'collision_per_step':is_colliding_per_step,
            'time_to_collision':times_to_collision,
            'distance_to_road_edge':distances_to_road_edge,
            'offroad_per_step':is_offroad_per_step,
            'speed_validity':speed_validity,
            'acceleration_validity':acceleration_validity,
            'traffic_light_violation_per_step':red_light_violations,
    }



def compute_scenario_rollouts_features(
        gt_scenario: dict,
        scenario_rollouts: dict,
        version: str,
) -> tuple[dict, dict]:
    """Computes the metrics features for both logged and simulated scenarios.

    Args:
        scenario: The `Scenario` loaded from WOMD.
        scenario_rollouts: The collection of joint scenes from simulation.

    Returns:
        Two `MetricFeatures`, the first one from logged data with n_samples=1 and
        the second from simulation with n_samples=`submission_specs.N_ROLLOUTS`.
    """

    #assert (gt_scenario['sim_agent_index'] == torch.tensor(scenario_rollouts['agent_id'])).all()
    all_agent_ids = gt_scenario['object_ids']
    object_type = gt_scenario['object_types']
    all_sim_agent_ids = gt_scenario['sim_agent_ids']
    evaluated_sim_agent_ids = gt_scenario['predict_agent_ids']
    pred_agent_ids = scenario_rollouts['agent_id']
    rollout_trajectories = scenario_rollouts['simulated_states']
    gt_trajectories = gt_scenario['tracks']
    traffic_signals = gt_scenario['traffic_signals']
    lane_ids = gt_scenario['lane_ids']
    lane_polylines = gt_scenario['lane_polylines']
    non_evaluated_sim_agent_ids = all_sim_agent_ids[~torch.isin(all_sim_agent_ids, evaluated_sim_agent_ids)]
    all_sim_agent_ids = torch.cat([evaluated_sim_agent_ids, non_evaluated_sim_agent_ids])
    _, pred2_all_sim_indices = torch.where(all_sim_agent_ids.unsqueeze(1) == pred_agent_ids.unsqueeze(0))
    _, pred2_val_sim_indices = torch.where(evaluated_sim_agent_ids.unsqueeze(1) == pred_agent_ids.unsqueeze(0))
    _, gt2_all_sim_indices = torch.where(all_sim_agent_ids.unsqueeze(1) == all_agent_ids.unsqueeze(0))
    _, gt2_val_sim_indices = torch.where(evaluated_sim_agent_ids.unsqueeze(1) == all_agent_ids.unsqueeze(0))
    evaluated_sim_agent_type = object_type[gt2_val_sim_indices]
    simulated_all_trajectories = rollout_trajectories[:, pred2_all_sim_indices]
    simulated_val_trajectories = rollout_trajectories[:, pred2_val_sim_indices]
    logged_all_trajectories = gt_trajectories[gt2_all_sim_indices]
    logged_all_trajectorie_masks = gt_scenario['track_masks'][gt2_all_sim_indices]
    logged_val_trajectories = gt_trajectories[gt2_val_sim_indices]
    logged_val_trajectorie_masks = gt_scenario['track_masks'][gt2_val_sim_indices]
    evaluated_object_mask = torch.isin(all_sim_agent_ids, evaluated_sim_agent_ids)
    log_features = compute_metric_features(
        all_agent_ids, evaluated_sim_agent_type, logged_all_trajectories.unsqueeze(0), logged_val_trajectories.unsqueeze(0), logged_val_trajectories, logged_val_trajectorie_masks, logged_all_trajectories, logged_all_trajectorie_masks, evaluated_object_mask, gt_scenario['road_edges'],True, lane_ids, lane_polylines, traffic_signals, version)
    segment_num = 32
    sim_features = []
    for i in range(32//segment_num):
        sim_features.append(compute_metric_features(
            all_agent_ids, evaluated_sim_agent_type, simulated_all_trajectories[i*segment_num:(i+1)*segment_num], simulated_val_trajectories[i*segment_num:(i+1)*segment_num], logged_val_trajectories, logged_val_trajectorie_masks, logged_all_trajectories, logged_all_trajectorie_masks, evaluated_object_mask, gt_scenario['road_edges'],False, lane_ids, lane_polylines, traffic_signals, version)) 
    all_sim_feature = {
        'average_displacement_error':[],
        'linear_speed':[],
        'linear_acceleration':[],
        'angular_speed':[],
        'angular_acceleration':[],
        'distance_to_nearest_object':[],
        'collision_per_step':[],
        'time_to_collision':[],
        'distance_to_road_edge':[],
        'offroad_per_step':[],
        'traffic_light_violation_per_step':[],
    }
    for sim_feature in sim_features:
        for key in all_sim_feature.keys():
            all_sim_feature[key].append(sim_feature[key])
    all_sim_feature = {key:torch.cat(value,dim=0) for key, value in all_sim_feature.items()}
    all_sim_feature['speed_validity'] = sim_features[0]['speed_validity']
    all_sim_feature['acceleration_validity'] = sim_features[0]['acceleration_validity']

    return log_features, all_sim_feature, logged_val_trajectorie_masks[:,11:]
