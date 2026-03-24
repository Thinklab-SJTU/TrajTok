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

import pickle
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional

from torch_geometric.data import Dataset

from src.utils import RankedLogger
import numpy as np
import torch

log = RankedLogger(__name__, rank_zero_only=True)

class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        gt_scenario_dir: Optional[str] = None,
        random_scene_scale_config: Optional[dict] = None,
        random_time_shift_config: Optional[dict] = None,
        random_time_mask_config: Optional[dict] = None,
        random_scene_crop_config: Optional[dict] = None,
    ) -> None:
        #raw_dir = Path(raw_dir)
        self._raw_paths = [os.path.join(raw_dir, p) for p in os.listdir(raw_dir)]

        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        self._gt_scenario_dir = (
            Path(gt_scenario_dir) if gt_scenario_dir is not None else None
        )

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )
        self.random_scene_scale_config = random_scene_scale_config
        self.random_time_shift_config = random_time_shift_config
        self.random_time_mask_config = random_time_mask_config
        self.random_scene_crop_config = random_scene_crop_config

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        if self._gt_scenario_dir is not None:
            gt_path = self._gt_scenario_dir / f"{data['scenario_id']}.pkl"
            with open(gt_path, "rb") as handle:
                data["gt_scenario"] = SimpleNamespace(value=pickle.load(handle))
        if self.random_scene_scale_config is not None:
            data = self.random_scene_scale(self.random_scene_scale_config, data)
        if self.random_time_shift_config is not None:
            data = self.random_time_shift(self.random_time_shift_config, data)
        if self.random_time_mask_config is not None:
            data = self.random_time_mask(self.random_time_mask_config, data)
        if self.random_scene_crop_config is not None:
            data = self.random_scene_crop(self.random_scene_crop_config, data)

        return data

    def random_scene_scale(self, config, data):
        scale_range = config['SCALE_RANGE']

        scale = np.random.uniform(scale_range[0], scale_range[1])
        data['map_save']['traj_pos'] *= scale
        data['agent']['position'][:,:,0:2] *= scale
        data['agent']['velocity'][:,:,0:2] *= scale
        return data
    
    def random_time_shift(self, config, data):
        max_time_shift = config['MAX_TIME_SHIFT']
        track_to_predict = data['agent']['role'][:,2]
        valid_time_mask = data['agent']['valid_mask'][track_to_predict][:, 10-max_time_shift:10+max_time_shift]
        valid_time_offset = valid_time_mask.all(dim=0).nonzero().reshape(-1)
        time_shift = np.random.choice(valid_time_offset) - max_time_shift
        if time_shift > 0:
            data['agent']['position'][:, :-time_shift, :] = data['agent']['position'][:, time_shift:, :].clone()
            data['agent']['velocity'][:, :-time_shift, :] = data['agent']['velocity'][:, time_shift:, :].clone()
            data['agent']['heading'][:, :-time_shift] = data['agent']['heading'][:, time_shift:].clone()
            data['agent']['position'][:, -time_shift:, :] = 0
            data['agent']['velocity'][:, -time_shift:, :] = 0
            data['agent']['heading'][:, -time_shift:] = 0
            data['agent']['valid_mask'][:, -time_shift:] = False
        elif time_shift < 0:
            time_shift = abs(time_shift)
            data['agent']['position'][:, time_shift:, :] = data['agent']['position'][:, :-time_shift, :].clone()
            data['agent']['velocity'][:, time_shift:, :] = data['agent']['velocity'][:, :-time_shift, :].clone()
            data['agent']['heading'][:, time_shift:] = data['agent']['heading'][:, :-time_shift].clone()
            data['agent']['position'][:, :time_shift, :] = 0
            data['agent']['velocity'][:, :time_shift, :] = 0
            data['agent']['heading'][:, :time_shift] = 0
            data['agent']['valid_mask'][:, :time_shift] = False

        return data
    
    def random_time_mask(self, config, data):
        time_mask_prob = config['TIME_MASK_PROB']
        valid_mask = data['agent']['valid_mask']
        num_objects, num_timestamp = valid_mask.shape
        rand_time_mask =  torch.rand(num_objects, num_timestamp) > time_mask_prob
        rand_time_mask[:, 10] = True
        new_mask = rand_time_mask & valid_mask
        data['agent']['valid_mask'] = new_mask
        data['agent']['position'][~new_mask] = 0
        data['agent']['velocity'][~new_mask] = 0
        data['agent']['heading'][~new_mask] = 0

        return data
    
    def random_scene_crop(self, config, data):
        max_dist = config['MAX_DIST']
        track_index_to_predict = torch.nonzero(data['agent']['role'][:,2]).reshape(-1)
        obj_pos = data['agent']['position']
        valid_mask = data['agent']['valid_mask']
        obj_last_pos = torch.ones((obj_pos.shape[0], 2)) * 0x7FFFFFFF
        for k in range(11):
            cur_valid_mask = valid_mask[:, k]  # (num_objects)
            obj_last_pos[cur_valid_mask] = obj_pos[:, k, 0:2][cur_valid_mask]
        center_pos = obj_last_pos[np.random.choice(track_index_to_predict)]
        dist = torch.norm(center_pos.unsqueeze(0) - obj_last_pos, dim=-1)
        crop_valid_mask = dist < max_dist
        crop_valid_mask[track_index_to_predict] = 1
        crop_valid_mask[-1] = 1 
        new_mask = crop_valid_mask.unsqueeze(1) & valid_mask
        data['agent']['valid_mask'] = new_mask
        data['agent']['position'][~new_mask] = 0
        data['agent']['velocity'][~new_mask] = 0
        data['agent']['heading'][~new_mask] = 0
        
        return data
