from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.protos import map_pb2

_LaneType = map_pb2.LaneCenter.LaneType

def extract_gt_scenario(scenario: scenario_pb2.Scenario, device='cpu') -> Dict:
    num_tracks = len(scenario.tracks)
    num_steps = 91
    tracks = torch.zeros(num_tracks, num_steps, 9, device=device)
    track_masks = torch.zeros(num_tracks, num_steps, dtype=torch.bool, device=device)
    object_ids = torch.zeros(num_tracks, device=device)
    object_types = torch.zeros(num_tracks, device=device)
    predict_index = {scenario.sdc_track_index}
    difficulty = []
    for track_idx, track in enumerate(scenario.tracks):
        for state_idx, state in enumerate(track.states):
            tracks[track_idx, state_idx, :] = torch.tensor([
                state.center_x,
                state.center_y,
                state.center_z,
                state.length,
                state.width,
                state.height,
                state.heading,
                state.velocity_x,
                state.velocity_y
            ], device=device)
            track_masks[track_idx, state_idx] = state.valid
            object_ids[track_idx] = track.id
            object_types[track_idx] = track.object_type
    tracks[:, 11:, 3:6] = tracks[:, 10, 3:6].unsqueeze(1)
        
    for required_prediction in scenario.tracks_to_predict:
        predict_index.add(required_prediction.track_index)
        #difficulty.append(required_prediction.difficulty)

    predict_index = torch.tensor(list(predict_index), device=device)
    #difficulty = torch.tensor(difficulty, device=device)

    road_edges =[]
    lane_ids = []
    lane_polylines = []
    for map_feature in scenario.map_features:
        if map_feature.HasField('road_edge'):
            polyline = []
            for point in map_feature.road_edge.polyline:
                polyline.append([point.x, point.y, point.z])
            polyline = torch.tensor(polyline, device=device)
            road_edges.append(polyline)
        if map_feature.HasField('lane'):
            if map_feature.lane.type == _LaneType.TYPE_SURFACE_STREET:
                lane_ids.append(map_feature.id)
                lane_polylines.append(list(map_feature.lane.polyline))
    dynamic_map_states = scenario.dynamic_map_states
    traffic_signals = []
    for dynamic_map_state in dynamic_map_states:
        traffic_signals.append(list(dynamic_map_state.lane_states))
    
    return {'scenario_id': scenario.scenario_id,
            'timestamps_seconds': list(scenario.timestamps_seconds),
            'current_time_index': scenario.current_time_index,
            'sdc_track_index': scenario.sdc_track_index,
            'objects_of_interest': list(scenario.objects_of_interest),
            'tracks': tracks,
            'track_masks': track_masks,
            'object_ids': object_ids.int(),
            'object_types': object_types,
            'road_edges': road_edges,
            'predict_index': predict_index.int(),
            'sim_agent_ids': torch.tensor(submission_specs.get_sim_agent_ids(scenario, submission_specs.ChallengeType.SIM_AGENTS), device=device).int(),
            'predict_agent_ids':  torch.sort(object_ids[predict_index])[0].int(),
            'lane_ids': lane_ids,
            'lane_polylines': lane_polylines,
            'traffic_signals': traffic_signals,
            }
    
def gt_scenario_to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        for i in range(len(x)):
            x[i] = gt_scenario_to_device(x[i], device)
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = gt_scenario_to_device(v, device)
    return x



