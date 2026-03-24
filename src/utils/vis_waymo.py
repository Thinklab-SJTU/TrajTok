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

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import cv2, pickle
import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2

from .video_recorder import ImageEncoder

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_VIOLET = (170, 0, 255)
COLOR_BUTTER = (252, 233, 79)
COLOR_ORANGE = (209, 92, 0)
COLOR_CHOCOLATE = (143, 89, 2)
COLOR_CHAMELEON = (78, 154, 6)
COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (32, 74, 135)
COLOR_PLUM = (92, 53, 102)
COLOR_SCARLET_RED = (164, 0, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (66, 62, 64)


class VisWaymo:
    def __init__(
        self,
        scenario_path: str,
        save_dir: Path,
        px_per_m: float = 10.0,
        video_size: int = 960,
        n_step: int = 91,
        step_current: int = 10,
        vis_ghost_gt: bool = True,
    ) -> None:
        self.px_per_m = px_per_m
        self.video_size = video_size
        self.n_step = n_step
        self.step_current = step_current
        self.px_agent2bottom = video_size // 2
        self.vis_ghost_gt = vis_ghost_gt

        # colors
        self.lane_style = [
            (COLOR_WHITE, 6),  # FREEWAY = 0
            (COLOR_ALUMINIUM_2, 6),  # SURFACE_STREET = 1
            (COLOR_ORANGE, 6),  # STOP_SIGN = 2
            (COLOR_CHOCOLATE, 6),  # BIKE_LANE = 3
            (COLOR_SKY_BLUE_1, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_BUTTER, 2),  # BROKEN = 6
            (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
            (COLOR_SCARLET_RED, 2),  # DOUBLE = 8
            (COLOR_CHAMELEON, 4),  # SPEED_BUMP = 9
            (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        ]

        self.tl_style = [
            COLOR_ALUMINIUM_1,  # STATE_UNKNOWN = 0;
            COLOR_RED,  # STOP = 1;
            COLOR_YELLOW,  # CAUTION = 2;
            COLOR_GREEN,  # GO = 3;
            COLOR_VIOLET,  # FLASHING = 4;
        ]
        # sdc=0, interest=1, predict=2
        self.agent_role_style = [COLOR_CYAN, COLOR_CHAMELEON, COLOR_MAGENTA]

        self.agent_cmd_txt = [
            "STATIONARY",  # STATIONARY = 0;
            "STRAIGHT",  # STRAIGHT = 1;
            "STRAIGHT_LEFT",  # STRAIGHT_LEFT = 2;
            "STRAIGHT_RIGHT",  # STRAIGHT_RIGHT = 3;
            "LEFT_U_TURN",  # LEFT_U_TURN = 4;
            "LEFT_TURN",  # LEFT_TURN = 5;
            "RIGHT_U_TURN",  # RIGHT_U_TURN = 6;
            "RIGHT_TURN",  # RIGHT_TURN = 7;
        ]

        # load tfrecord scenario
        
        with open(scenario_path,'rb') as f: 
            scenario = pickle.load(f)
        
        
        # scenario = scenario_pb2.Scenario()
        # for data in tf.data.TFRecordDataset([scenario_path], compression_type=""):
        #     scenario.ParseFromString(bytes(data.numpy()))
        #     break

        # make output dir
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # draw gt
        mp_xyz, mp_id, mp_type = get_map_features(scenario.map_features)

        tl_lane_state, tl_lane_id = get_traffic_light_features(
            scenario.dynamic_map_states
        )
        ag_valid, ag_xy, ag_yaw, ag_size, ag_role, ag_id = get_agent_features(
            scenario, step_current=step_current
        )
        self.ag_id2size = dict(zip(ag_id, ag_size))
        self.ag_id2role = dict(zip(ag_id, ag_role))

        raster_map, self.top_left_px = self._register_map(mp_xyz, self.px_per_m)
        self._draw_map(raster_map, mp_xyz, mp_type)

        im_gt_maps = [raster_map.copy() for _ in range(n_step)]
        self._draw_traffic_lights(im_gt_maps, tl_lane_state, tl_lane_id, mp_xyz, mp_id)

        # save gt video and get paths for wandb logging
        im_gt = deepcopy(im_gt_maps)
        self._draw_agents(im_gt, ag_valid, ag_xy, ag_yaw, ag_size, ag_role)

        gt_video_path = (self.save_dir / "gt.mp4").as_posix()
        save_images_to_mp4(im_gt, gt_video_path)
        self.video_paths = [gt_video_path]

        # prepare images for drawing prediction on top
        self.im_gt_blended = []
        if self.vis_ghost_gt:
            im_gt_agents = [np.zeros_like(raster_map) for _ in range(n_step)]
            self._draw_agents(im_gt_agents, ag_valid, ag_xy, ag_yaw, ag_size, ag_role)
            for i in range(n_step):
                self.im_gt_blended.append(
                    cv2.addWeighted(im_gt_agents[i], 0.6, im_gt_maps[i], 1, 0)
                )
        else:
            for i in range(n_step):
                if i <= 10:
                    self.im_gt_blended.append(deepcopy(im_gt[i]))
                else:
                    self.im_gt_blended.append(deepcopy(im_gt_maps[i]))

    @staticmethod
    def _register_map(
        mp_xyz: List[np.ndarray], px_per_m: float, edge_px: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            mp_xyz:  len=n_pl, list of np array [n_pl_node, 3]
            px_per_m: float

        Returns:
            raster_map: empty image
            top_left_px
        """
        xmin = min([arr[:, 0].min() for arr in mp_xyz])
        xmax = max([arr[:, 0].max() for arr in mp_xyz])
        ymin = min([arr[:, 1].min() for arr in mp_xyz])
        ymax = max([arr[:, 1].max() for arr in mp_xyz])
        map_boundary = np.array([xmin, xmax, ymin, ymax])

        # y axis is inverted in pixel coordinate
        xmin, xmax, ymax, ymin = (map_boundary * px_per_m).astype(np.int64)
        ymax *= -1
        ymin *= -1
        xmin -= edge_px
        ymin -= edge_px
        xmax += edge_px
        ymax += edge_px

        raster_map = np.zeros([ymax - ymin, xmax - xmin, 3], dtype=np.uint8)
        top_left_px = np.array([xmin, ymin], dtype=np.float32)
        return raster_map, top_left_px

    def _draw_map(
        self, raster_map: np.ndarray, mp_xyz: List[np.ndarray], mp_type: np.ndarray
    ) -> None:
        """
        Args: numpy arrays
            mp_xyz:  len=n_pl, list of np array [n_pl_node, 3]
            mp_type: [n_pl], int

        Returns:
            draw on raster_map
        """
        for i, _type in enumerate(mp_type):
            color, thickness = self.lane_style[_type]
            cv2.polylines(
                raster_map,
                [self._to_pixel(mp_xyz[i][:, :2])],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_traffic_lights(
        self,
        input_images: List[np.ndarray],
        tl_lane_state: List[np.ndarray],
        tl_lane_id: List[np.ndarray],
        mp_xyz: List[np.ndarray],
        mp_id: np.ndarray,
    ) -> None:
        for step_t, step_image in enumerate(input_images):
            if step_t < len(tl_lane_state):
                for i_tl, _state in enumerate(tl_lane_state[step_t]):
                    _lane_id = tl_lane_id[step_t][i_tl]
                    _lane_idx = np.argwhere(mp_id == _lane_id).item()
                    pos = self._to_pixel(mp_xyz[_lane_idx][:, :2])
                    cv2.polylines(
                        step_image,
                        [pos],
                        isClosed=False,
                        color=self.tl_style[_state],
                        thickness=8,
                        lineType=cv2.LINE_AA,
                    )
                    if _state >= 1 and _state <= 3:
                        cv2.drawMarker(
                            step_image,
                            pos[-1],
                            color=self.tl_style[_state],
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=10,
                            thickness=6,
                        )

    def _draw_agents(
        self,
        input_images: List[np.ndarray],
        ag_valid: np.ndarray,  # [n_ag, n_step], bool
        ag_xy: np.ndarray,  # [n_ag, n_step, 2], (x,y)
        ag_yaw: np.ndarray,  # [n_ag, n_step, 1], [-pi, pi]
        ag_size: np.ndarray,  # [n_ag, 3], [length, width, height]
        ag_role: np.ndarray,  # [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
    ) -> None:
        for step_t, step_image in enumerate(input_images):
            if step_t < ag_valid.shape[1]:
                _valid = ag_valid[:, step_t]  # [n_ag]
                _pos = ag_xy[:, step_t]  # [n_ag, 2]
                _yaw = ag_yaw[:, step_t]  # [n_ag, 1]

                bbox_gt = self._to_pixel(
                    self._get_agent_bbox(_valid, _pos, _yaw, ag_size)
                )
                heading_start = self._to_pixel(_pos[_valid])
                _yaw = _yaw[:, 0][_valid]
                heading_end = self._to_pixel(
                    _pos[_valid] + 1.5 * np.stack([np.cos(_yaw), np.sin(_yaw)], axis=-1)
                )
                _role = ag_role[_valid]
                for i in range(_role.shape[0]):
                    if not _role[i].any():
                        color = COLOR_ALUMINIUM_0
                    else:
                        color = self.agent_role_style[np.where(_role[i])[0].min()]
                    cv2.fillConvexPoly(step_image, bbox_gt[i], color=color)
                    cv2.arrowedLine(
                        step_image,
                        heading_start[i],
                        heading_end[i],
                        color=COLOR_BLACK,
                        thickness=4,
                        line_type=cv2.LINE_AA,
                        tipLength=0.6,
                    )

    def save_video_scenario_rollout(
        self,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
        n_vis_rollout: int,
    ):
        for i_rollout in range(n_vis_rollout):
            images = deepcopy(self.im_gt_blended)
            ag_valid, ag_xy, ag_yaw, ag_size, ag_role = self._get_features_from_trajs(
                scenario_rollout.joint_scenes[i_rollout].simulated_trajectories
            )
            self._draw_agents(
                images[self.step_current + 1 :],
                ag_valid,
                ag_xy,
                ag_yaw,
                ag_size,
                ag_role,
            )
            _video_path = (self.save_dir / f"rollout_{i_rollout:02d}.mp4").as_posix()
            self.video_paths.append(_video_path)
            save_images_to_mp4(images, _video_path)

    def _get_features_from_trajs(
        self, trajs: List[sim_agents_submission_pb2.SimulatedTrajectory]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ag_valid: [n_ag, n_step], bool
        ag_xy: [n_ag, n_step, 2], (x,y)
        ag_yaw: [n_ag, n_step, 1], [-pi, pi]
        ag_size: [n_ag, 3], [length, width, height]
        ag_role: [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
        """
        n_ag = len(trajs)
        n_step = len(trajs[0].center_x)
        ag_valid = np.ones([n_ag, n_step], dtype=bool)
        ag_xy = np.zeros([n_ag, n_step, 2], dtype=np.float32)
        ag_yaw = np.zeros([n_ag, n_step, 1], dtype=np.float32)
        ag_size = np.zeros([n_ag, 3], dtype=np.float32)
        ag_role = np.zeros([n_ag, 3], dtype=bool)

        for i_ag, _traj in enumerate(trajs):
            ag_xy[i_ag] = np.stack([_traj.center_x, _traj.center_y], axis=-1)
            ag_yaw[i_ag, :, 0] = _traj.heading
            ag_size[i_ag] = self.ag_id2size[_traj.object_id]
            ag_role[i_ag] = self.ag_id2role[_traj.object_id]

        return ag_valid, ag_xy, ag_yaw, ag_size, ag_role

    def _to_pixel(self, pos: np.ndarray) -> np.ndarray:
        pos = pos * self.px_per_m
        pos[..., 0] = pos[..., 0] - self.top_left_px[0]
        pos[..., 1] = -pos[..., 1] - self.top_left_px[1]
        return np.round(pos).astype(np.int32)

    @staticmethod
    def _get_agent_bbox(
        agent_valid: np.ndarray,
        agent_pos: np.ndarray,
        agent_yaw: np.ndarray,
        agent_size: np.ndarray,
    ) -> np.ndarray:
        yaw = agent_yaw[agent_valid]  # n, 1
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        v_forward = np.concatenate([cos_yaw, sin_yaw], axis=-1)  # n,2
        v_right = np.concatenate([sin_yaw, -cos_yaw], axis=-1)

        offset_forward = 0.5 * agent_size[agent_valid, 0:1] * v_forward  # [n, 2]
        offset_right = 0.5 * agent_size[agent_valid, 1:2] * v_right  # [n, 2]

        vertex_offset = np.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            axis=1,
        )  # n,4,2

        agent_pos = agent_pos[agent_valid]
        bbox = agent_pos[:, None, :].repeat(4, 1) + vertex_offset  # n,4,2
        return bbox


def save_images_to_mp4(images: List[np.ndarray], out_path: str, fps=20) -> None:
    encoder = ImageEncoder(out_path, images[0].shape, fps, fps)
    for im in images:
        encoder.capture_frame(im)
    encoder.close()
    encoder = None


def get_agent_features(
    scenario: scenario_pb2.Scenario, step_current: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ag_valid: [n_ag, n_step], bool
    ag_xy: [n_ag, n_step, 2], (x,y)
    ag_yaw: [n_ag, n_step, 1], [-pi, pi]
    ag_size: [n_ag, 3], [length, width, height]
    ag_role: [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
    ag_id: [n_ag], int
    """
    tracks = scenario.tracks
    sdc_track_index = scenario.sdc_track_index
    track_index_predict = ([i.track_index for i in scenario.tracks_to_predict],)
    object_id_interest = ([i for i in scenario.objects_of_interest],)

    ag_valid, ag_xy, ag_yaw, ag_size, ag_role, ag_id = [], [], [], [], [], []
    for i, _track in enumerate(tracks):
        # [VEHICLE=1, PEDESTRIAN=2, CYCLIST=3] -> [0,1,2]
        # ag_type.append(_track.object_type - 1)
        if _track.states[step_current].valid:
            ag_id.append(_track.id)
            step_valid, step_xy, step_yaw = [], [], []
            for s in _track.states:
                step_valid.append(s.valid)
                step_xy.append([s.center_x, s.center_y])
                step_yaw.append([s.heading])

            ag_valid.append(step_valid)
            ag_xy.append(step_xy)
            ag_yaw.append(step_yaw)

            ag_size.append(
                [
                    _track.states[step_current].length,
                    _track.states[step_current].width,
                    _track.states[step_current].height,
                ]
            )

            ag_role.append([False, False, False])
            if i in track_index_predict:
                ag_role[-1][2] = True
            if _track.id in object_id_interest:
                ag_role[-1][1] = True
            if i == sdc_track_index:
                ag_role[-1][0] = True

    ag_valid = np.array(ag_valid)
    ag_xy = np.array(ag_xy)
    ag_yaw = np.array(ag_yaw)
    ag_size = np.array(ag_size)
    ag_role = np.array(ag_role)
    ag_id = np.array(ag_id)
    return ag_valid, ag_xy, ag_yaw, ag_size, ag_role, ag_id


def get_traffic_light_features(
    tl_features,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """n_tl is not constant for each timestep
    tl_lane_state: len=n_step, list of array [n_tl]
    tl_lane_id: len=n_step, list of array [n_tl]
    """
    tl_lane_state, tl_lane_id, tl_stop_point = [], [], []
    for _step_tl in tl_features:
        step_tl_lane_state, step_tl_lane_id, step_tl_stop_point = [], [], []
        for _tl in _step_tl.lane_states:  # modify LANE_STATE
            if _tl.state == 0:  # UNKNOWN = 0;
                tl_state = 0  # UNKNOWN = 0;
            elif _tl.state in [1, 4]:  # ARROW_STOP = 1; STOP = 4;
                tl_state = 1  # STOP = 1;
            elif _tl.state in [2, 5]:  # ARROW_CAUTION = 2; CAUTION = 5;
                tl_state = 2  # CAUTION = 2;
            elif _tl.state in [3, 6]:  # ARROW_GO = 3; GO = 6;
                tl_state = 3  # GO = 3;
            elif _tl.state in [7, 8]:  # FLASHING_STOP = 7; FLASHING_CAUTION = 8;
                tl_state = 4  # FLASHING = 4;
            else:
                assert ValueError

            step_tl_lane_state.append(tl_state)
            step_tl_lane_id.append(_tl.lane)

        tl_lane_state.append(np.array(step_tl_lane_state))
        tl_lane_id.append(np.array(step_tl_lane_id))
    return tl_lane_state, tl_lane_id


def get_map_features(
    map_features,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    mp_xyz, mp_id, mp_type = [], [], []
    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        # pip install waymo-open-dataset-tf-2-6-0==1.4.9, not updated, should be driveway
        if feature_data_type is None:
            continue
        feature = getattr(mf, feature_data_type)
        if feature_data_type == "lane":
            if feature.type == 0:  # UNDEFINED
                mp_type.append(1)
            elif feature.type == 1:  # FREEWAY
                mp_type.append(0)
            elif feature.type == 2:  # SURFACE_STREET
                mp_type.append(1)
            elif feature.type == 3:  # BIKE_LANE
                mp_type.append(3)
            mp_id.append(mf.id)
            mp_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])
        elif feature_data_type == "stop_sign":
            for l_id in feature.lane:
                # override FREEWAY/SURFACE_STREET with stop sign lane
                # BIKE_LANE remains unchanged
                idx_lane = mp_id.index(l_id)
                if mp_type[idx_lane] < 2:
                    mp_type[idx_lane] = 2
        elif feature_data_type == "road_edge":
            assert feature.type > 0  # no UNKNOWN = 0
            mp_id.append(mf.id)
            mp_type.append(feature.type + 3)  # [1, 2] -> [4, 5]
            mp_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])
        elif feature_data_type == "road_line":
            assert feature.type > 0  # no UNKNOWN = 0
            # BROKEN_SINGLE_WHITE = 1
            # SOLID_SINGLE_WHITE = 2
            # SOLID_DOUBLE_WHITE = 3
            # BROKEN_SINGLE_YELLOW = 4
            # BROKEN_DOUBLE_YELLOW = 5
            # SOLID_SINGLE_YELLOW = 6
            # SOLID_DOUBLE_YELLOW = 7
            # PASSING_DOUBLE_YELLOW = 8
            if feature.type in [1, 4, 5]:
                feature_type_new = 6  # BROKEN
            elif feature.type in [2, 6]:
                feature_type_new = 7  # SOLID_SINGLE
            else:
                feature_type_new = 8  # DOUBLE
            mp_id.append(mf.id)
            mp_type.append(feature_type_new)
            mp_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])
        elif feature_data_type in ["speed_bump", "driveway", "crosswalk"]:
            xyz = np.array([[p.x, p.y, p.z] for p in feature.polygon])
            polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)
            pl_polygon = _get_polylines_from_polygon(xyz[polygon_idx])
            mp_xyz.extend(pl_polygon)
            mp_id.extend([mf.id] * len(pl_polygon))
            pl_type = 9 if feature_data_type in ["speed_bump", "driveway"] else 10
            mp_type.extend([pl_type] * len(pl_polygon))
        else:
            raise ValueError

    mp_id = np.array(mp_id)  # [n_pl]
    mp_type = np.array(mp_type)  # [n_pl]
    mp_xyz = [np.stack(line) for line in mp_xyz]  # len=n_pl, list of [n_pl_node, 3]
    return mp_xyz, mp_id, mp_type


def _get_polylines_from_polygon(polygon: np.ndarray) -> List[List[List]]:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> List[List]:
        length = np.linalg.norm(start - end)
        unit_vec = (end - start) / length
        pl = []
        for i in range(int(length) + 1):  # 4.5 -> 5 [0,1,2,3,4]
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])
        pl.append([end[0], end[1], end[2]])
        return pl

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])
    return [pl1, pl1[::-1], pl2, pl2[::-1]]
