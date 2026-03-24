
import multiprocessing
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from src.smart.utils.geometry import wrap_angle
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map
from wosac_fast_eval_tool.scenario_gt_converter import extract_gt_scenario

_polygon_types = ["lane", "road_edge", "road_line", "crosswalk"]
_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]


def get_agent_features(
    track_infos: Dict[str, np.ndarray], split, num_historical_steps, num_steps
) -> Dict[str, Any]:
    """
    track_infos:
    object_id (100,) int64
    object_type (100,) uint8
    states (100, 91, 9) float32
    valid (100, 91) bool
    role (100, 3) bool
    """

    idx_agents_to_add = []
    for i in range(len(track_infos["object_id"])):
        add_agent = track_infos["valid"][i, num_historical_steps - 1]

        if add_agent:
            idx_agents_to_add.append(i)

    num_agents = len(idx_agents_to_add)
    out_dict = {
        "num_nodes": num_agents,
        "valid_mask": torch.zeros([num_agents, num_steps], dtype=torch.bool),
        "role": torch.zeros([num_agents, 3], dtype=torch.bool),
        "id": torch.zeros(num_agents, dtype=torch.int64) - 1,
        "type": torch.zeros(num_agents, dtype=torch.uint8),
        "position": torch.zeros([num_agents, num_steps, 3], dtype=torch.float32),
        "heading": torch.zeros([num_agents, num_steps], dtype=torch.float32),
        "velocity": torch.zeros([num_agents, num_steps, 2], dtype=torch.float32),
        "shape": torch.zeros([num_agents, 3], dtype=torch.float32),
    }

    for i, idx in enumerate(idx_agents_to_add):

        out_dict["role"][i] = torch.from_numpy(track_infos["role"][idx])
        out_dict["id"][i] = track_infos["object_id"][idx]
        out_dict["type"][i] = track_infos["object_type"][idx]

        valid = track_infos["valid"][idx] # [n_step]
        states = track_infos["states"][idx]
        #print(states.shape)

        object_shape = states[:, 3:6]  # [n_step, 3], length, width, height
        object_shape = object_shape[valid].mean(axis=0)  # [3]
        out_dict["shape"][i] = torch.from_numpy(object_shape)

        valid_steps = np.where(valid)[0]
        position = states[:, :3]  # [n_step, dim], x, y, z
        velocity = states[:, 7:9]  # [n_step, 2], vx, vy
        heading = states[:, 6]  # [n_step], heading
        if valid.sum() > 1:
            t_start, t_end = valid_steps[0], valid_steps[-1]
            f_pos = interp1d(valid_steps, position[valid], axis=0)
            f_vel = interp1d(valid_steps, velocity[valid], axis=0)
            f_yaw = interp1d(valid_steps, np.unwrap(heading[valid], axis=0), axis=0)
            t_in = np.arange(t_start, t_end + 1)
            out_dict["valid_mask"][i, t_start : t_end + 1] = True
            out_dict["position"][i, t_start : t_end + 1] = torch.from_numpy(f_pos(t_in))
            out_dict["velocity"][i, t_start : t_end + 1] = torch.from_numpy(f_vel(t_in))
            out_dict["heading"][i, t_start : t_end + 1] = torch.from_numpy(f_yaw(t_in))
        else:
            t = valid_steps[0]
            out_dict["valid_mask"][i, t] = True
            out_dict["position"][i, t] = torch.from_numpy(position[t])
            out_dict["velocity"][i, t] = torch.from_numpy(velocity[t])
            out_dict["heading"][i, t] = torch.tensor(heading[t])

    return out_dict


def get_map_features(map_infos, tf_current_light, dim=2):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]
    num_polygons = len(polygon_ids)

    # initialization
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_light_type = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    # point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons

    for _key in _polygon_types:
        for _seg in map_infos[_key]:
            _idx = polygon_ids.index(_seg["id"])
            centerline = map_infos["all_polylines"][
                _seg["polyline_index"][0] : _seg["polyline_index"][1]
            ]
            centerline = torch.from_numpy(centerline).float()
            polygon_type[_idx] = _polygon_types.index(_key)

            point_position[_idx] = centerline[:-1, :dim]
            center_vectors = centerline[1:] - centerline[:-1]
            # point_orientation[_idx] = torch.cat(
            #     [torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0
            # )
            point_type[_idx] = torch.full(
                (len(center_vectors),), _seg["type"], dtype=torch.uint8
            )

            if _key == "lane":
                res = tf_current_light[tf_current_light["lane_id"] == _seg["id"]]
                if len(res) != 0:
                    polygon_light_type[_idx] = _polygon_light_type.index(
                        res["state"].item()
                    )

    num_points = torch.tensor(
        [point.size(0) for point in point_position], dtype=torch.long
    )
    point_to_polygon_edge_index = torch.stack(
        [
            torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points),
        ],
        dim=0,
    )

    map_data = {
        "map_polygon": {},
        "map_point": {},
        ("map_point", "to", "map_polygon"): {},
    }
    map_data["map_polygon"]["num_nodes"] = num_polygons
    map_data["map_polygon"]["type"] = polygon_type
    map_data["map_polygon"]["light_type"] = polygon_light_type
    if len(num_points) == 0:
        map_data["map_point"]["num_nodes"] = 0
        map_data["map_point"]["position"] = torch.tensor([], dtype=torch.float)
        # map_data["map_point"]["orientation"] = torch.tensor([], dtype=torch.float)
        map_data["map_point"]["type"] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data["map_point"]["num_nodes"] = num_points.sum().item()
        map_data["map_point"]["position"] = torch.cat(point_position, dim=0)
        # map_data["map_point"]["orientation"] = wrap_angle(
        #     torch.cat(point_orientation, dim=0)
        # )
        map_data["map_point"]["type"] = torch.cat(point_type, dim=0)
    map_data["map_point", "to", "map_polygon"][
        "edge_index"
    ] = point_to_polygon_edge_index
    return map_data


def process_dynamic_map(dynamic_map_infos):
    lane_ids = dynamic_map_infos["lane_id"]
    tf_lights = []
    for t in range(len(lane_ids)):
        lane_id = lane_ids[t]
        time = np.ones_like(lane_id) * t
        state = dynamic_map_infos["state"][t]
        tf_light = np.concatenate([lane_id, time, state], axis=0)
        tf_lights.append(tf_light)
    tf_lights = np.concatenate(tf_lights, axis=1).transpose(1, 0)
    tf_lights = pd.DataFrame(data=tf_lights, columns=["lane_id", "time_step", "state"])
    tf_lights["time_step"] = tf_lights["time_step"].astype("int")
    tf_lights["lane_id"] = tf_lights["lane_id"].astype("int")
    tf_lights["state"] = tf_lights["state"].astype("str")
    tf_lights.loc[tf_lights["state"].str.contains("STOP"), ["state"]] = (
        "LANE_STATE_STOP"
    )
    tf_lights.loc[tf_lights["state"].str.contains("GO"), ["state"]] = "LANE_STATE_GO"
    tf_lights.loc[tf_lights["state"].str.contains("CAUTION"), ["state"]] = (
        "LANE_STATE_CAUTION"
    )
    tf_lights.loc[tf_lights["state"].str.contains("UNKNOWN"), ["state"]] = (
        "LANE_STATE_UNKNOWN"
    )
    return tf_lights


def decode_tracks_from_proto(scenario):
    sdc_track_index = scenario.sdc_track_index
    track_index_predict = [i.track_index for i in scenario.tracks_to_predict]
    object_id_interest = [i for i in scenario.objects_of_interest]

    track_infos = {
        "object_id": [],
        "object_type": [],
        "states": [],
        "valid": [],
        "role": [],
    }
    for i, cur_data in enumerate(scenario.tracks):  # number of objects

        step_state = []
        step_valid = []
        for s in cur_data.states:
            step_state.append(
                [
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.length,
                    s.width,
                    s.height,
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                ]
            )
            step_valid.append(s.valid)
            # This angle is normalized to [-pi, pi). The velocity vector in m/s

        track_infos["object_id"].append(cur_data.id)
        track_infos["object_type"].append(cur_data.object_type - 1)
        track_infos["states"].append(np.array(step_state, dtype=np.float32))
        track_infos["valid"].append(np.array(step_valid))

        track_infos["role"].append([False, False, False])
        if i in track_index_predict:
            track_infos["role"][-1][2] = True  # predict=2
        if cur_data.id in object_id_interest:
            track_infos["role"][-1][1] = True  # interest=1
        if i == sdc_track_index:  # ego_vehicle=0
            track_infos["role"][-1][0] = True

    track_infos["states"] = np.array(track_infos["states"], dtype=np.float32)
    track_infos["valid"] = np.array(track_infos["valid"], dtype=bool)
    track_infos["role"] = np.array(track_infos["role"], dtype=bool)
    track_infos["object_id"] = np.array(track_infos["object_id"], dtype=np.int64)
    track_infos["object_type"] = np.array(track_infos["object_type"], dtype=np.uint8)
    return track_infos


def decode_map_features_from_proto(map_features):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    point_cnt = 0
    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        # pip install waymo-open-dataset-tf-2-6-0==1.4.9, not updated, should be driveway
        if feature_data_type is None:
            continue

        feature = getattr(mf, feature_data_type)
        if feature_data_type == "lane":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                if feature.type == 0:  # UNDEFINED
                    cur_info["type"] = 1
                elif feature.type == 1:  # FREEWAY
                    cur_info["type"] = 0
                elif feature.type == 2:  # SURFACE_STREET
                    cur_info["type"] = 1
                elif feature.type == 3:  # BIKE_LANE
                    cur_info["type"] = 3

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["lane"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type == "road_edge":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                # assert feature.type > 0
                cur_info["type"] = feature.type + 3

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["road_edge"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type == "road_line":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                # there is no UNKNOWN = 0
                # BROKEN_SINGLE_WHITE = 1
                # SOLID_SINGLE_WHITE = 2
                # SOLID_DOUBLE_WHITE = 3
                # BROKEN_SINGLE_YELLOW = 4
                # BROKEN_DOUBLE_YELLOW = 5
                # SOLID_SINGLE_YELLOW = 6
                # SOLID_DOUBLE_YELLOW = 7
                # PASSING_DOUBLE_YELLOW = 8
                # assert feature.type > 0  # no UNKNOWN = 0
                if feature.type in [1, 4, 5]:
                    cur_info["type"] = 6  # BROKEN
                elif feature.type in [2, 6]:
                    cur_info["type"] = 7  # SOLID_SINGLE
                else:
                    cur_info["type"] = 8  # DOUBLE

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["road_line"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type in ["speed_bump", "driveway", "crosswalk"]:
            xyz = np.array([[p.x, p.y, p.z] for p in feature.polygon])
            polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)
            pl_polygon = get_polylines_from_polygon(xyz[polygon_idx])
            cur_info = {"id": mf.id, "type": 9}

            cur_polyline = np.stack(
                [
                    np.array([p[0], p[1], p[2], cur_info["type"], cur_info["id"]])
                    for p in pl_polygon
                ],
                axis=0,
            )

            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            map_infos["crosswalk"].append(cur_info)
            polylines.append(cur_polyline)
            point_cnt += len(cur_polyline)

    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        if feature_data_type == "stop_sign":
            feature = mf.stop_sign
            for l_id in feature.lane:
                # override FREEWAY/SURFACE_STREET with stop sign lane
                # BIKE_LANE remains unchanged
                is_found = False
                for _i in range(len(map_infos["lane"])):
                    if map_infos["lane"][_i]["id"] == l_id:
                        is_found = True
                        if map_infos["lane"][_i]["type"] < 2:
                            map_infos["lane"][_i]["type"] = 2
                # not necessary found, some stop sign lanes are for lane with length 1
                # assert is_found

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    signal_state = {
        0: "LANE_STATE_UNKNOWN",
        #  States for traffic signals with arrows.
        1: "LANE_STATE_ARROW_STOP",
        2: "LANE_STATE_ARROW_CAUTION",
        3: "LANE_STATE_ARROW_GO",
        #  Standard round traffic signals.
        4: "LANE_STATE_STOP",
        5: "LANE_STATE_CAUTION",
        6: "LANE_STATE_GO",
        #  Flashing light signals.
        7: "LANE_STATE_FLASHING_STOP",
        8: "LANE_STATE_FLASHING_CAUTION",
    }

    dynamic_map_infos = {"lane_id": [], "state": []}
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state = [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])

        dynamic_map_infos["lane_id"].append(np.array([lane_id]))
        dynamic_map_infos["state"].append(np.array([state]))

    return dynamic_map_infos


def wm2argo(file_path, split, output_dir, output_dir_gt):
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )
    for tf_data in dataset:
        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        track_infos = decode_tracks_from_proto(scenario)
        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(
            scenario.dynamic_map_states
        )

        current_time_index = scenario.current_time_index
        scenario_id = scenario.scenario_id
        tf_lights = process_dynamic_map(dynamic_map_infos)
        tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index]
        map_data = get_map_features(map_infos, tf_current_light)

        data = preprocess_map(map_data)
        data["agent"] = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
        )

        data["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data, f)

        if output_dir_gt is not None:
            gt_dict = extract_gt_scenario(scenario)
            with open(os.path.join(output_dir_gt, f"{scenario_id}.pkl"), 'wb') as f:
                pickle.dump(gt_dict, f)

def batch_process9s_transformer(input_dir, output_dir, split, num_workers):
    output_dir = Path(output_dir)
    if split == "validation":
        output_dir_gt = output_dir / "validation_gt"
        output_dir_gt.mkdir(exist_ok=True, parents=True)
    else:
        output_dir_gt = None
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)
    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
    func = partial(
        wm2argo,
        split=split,
        output_dir=output_dir,
        output_dir_gt=output_dir_gt,
    )

    with multiprocessing.Pool(num_workers) as p:
        r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir", type=str
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )
