import numpy as np
import pickle, os
from scipy.interpolate import CubicHermiteSpline
from ..utils import transform_to_local, wrap_angle, clean_heading
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

class TrajTok:
    def __init__(self):
        self.shift = 5
        self.t = 0.1 * self.shift
        self.agent_classes = ['veh', 'ped', 'cyc']
        self.flip_trajs = True
        # grid settings
        self.x_max = {'veh': 20, 'ped': 4.5, 'cyc': 8}
        self.x_min = {'veh': -5, 'ped': -1.5, 'cyc': -1}
        self.y_max = {'veh': 2, 'ped': 2, 'cyc': 1}
        self.y_min = {'veh': -2, 'ped': -2, 'cyc': -1}
        self.x_binnum = {'veh': 250, 'ped': 120, 'cyc': 180} # 0.05
        self.y_binnum = {'veh': 80, 'ped': 80, 'cyc': 40}  # 0.05
        # filter settings
        self.valid_count_threshold = {'veh': 6, 'ped': 1, 'cyc': 6}
        self.filter_range = {'veh': 4, 'ped': 2, 'cyc': 4}
        self.filter_threshold_add = {'veh': 20, 'ped': 2, 'cyc': 20}
        self.filter_threshold_remove = {'veh': 20, 'ped': 18, 'cyc': 20}
        # logged data extracting settings
        self.raw_data_path = 'data/waymo_processed/training'
        self.traj_data_path = 'data/waymo_processed/traj_data.pkl'
        self.max_workers = 16
        self.max_file_nums = 500000
        self.max_traj_nums = {'veh': 12000000, 'ped': 1490000, 'cyc': None}
        self.aggregation_chunk_size = 500000
        self.random_seed = 0
        self.rng = np.random.RandomState(self.random_seed)
        self.use_cache= True
        # output settings
        self.output_path = 'src/smart/tokens/trajtok_vocab.pkl'

        if self.use_cache and os.path.exists(self.traj_data_path):
            print(f"loading traj data cache from {self.traj_data_path}...")
            with open(self.traj_data_path, 'rb') as f:
                self.traj_data = pickle.load(f)
        else:
            self.get_traj_data_multi_workers()
            with open(self.traj_data_path, 'wb') as f:
                pickle.dump(self.traj_data, f)

    
    def get_traj_data_multi_workers(self):

        self.traj_data = {'veh': [], 'ped': [], 'cyc': []}

        file_names = os.listdir(self.raw_data_path)
        if self.max_file_nums:
            file_names = file_names[:self.max_file_nums]
        
        if self.max_workers == 0:
            for file in tqdm(file_names, desc="Extracting traj data"):
                result = self._get_traj_data(os.path.join(self.raw_data_path, file))
                for agent_class in self.agent_classes:
                    self.traj_data[agent_class].extend(result[agent_class])
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._get_traj_data, os.path.join(self.raw_data_path, file)) for file in file_names]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting traj data"):
                    try:
                        result = future.result()
                        for agent_class in self.agent_classes:
                            self.traj_data[agent_class].extend(result[agent_class])
                    except Exception as e:
                        print(f"Error extracting traj data: {e}")
        for agent_class in self.agent_classes:
            self.traj_data[agent_class] = torch.cat(self.traj_data[agent_class])
            headings = self.traj_data[agent_class][:,:,-1]
            heading_diffs = torch.abs(wrap_angle(headings[:,1:] - headings[:,:-1]))
            head_valid = heading_diffs.max(-1).values < 30 * np.pi/180
            self.traj_data[agent_class] = self.traj_data[agent_class][head_valid].numpy()
            print(f"traj num of {agent_class}: {len(self.traj_data[agent_class])}")
        
        

    def _get_traj_data(self, file_path):
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        n_agent, n_step, _ = data['agent']['position'].shape
        pos = data['agent']['position'][..., 0:2]
        masks = data['agent']['valid_mask']
        types = data['agent']['type']
        headings = wrap_angle(data['agent']['heading'])

        traj_data = {'veh': [], 'ped': [], 'cyc': []}

        for i in range(0, n_step-self.shift, self.shift):
            pos_local, head_local = transform_to_local(pos_global=pos[:, i+1:i+self.shift+1], 
                                                    head_global=headings[:, i+1:i+self.shift+1], 
                                                    pos_now=pos[:,i], 
                                                    head_now=headings[:,i])

            trajs = torch.cat([pos_local, head_local.unsqueeze(-1)], dim=-1)
            valid_mask = masks[:, i:i+self.shift+1].all(dim=-1)
            traj_data['veh'].append(trajs[(types==0) & valid_mask ])
            traj_data['ped'].append(trajs[(types==1) & valid_mask ])
            traj_data['cyc'].append(trajs[(types==2) & valid_mask ])
            
        return traj_data
        
    def cal_polygon_contour(
        self,
        pos,  # [n_agent, n_step, n_target, 2]
        head,  # [n_agent, n_step, n_target]
        width_length,  # [n_agent, 1, 1, 2]
    ) :  # [n_agent, n_step, n_target, 4, 2]
        x, y = pos[..., 0], pos[..., 1]  # [n_agent, n_step, n_target]
        width, length = width_length[..., 0], width_length[..., 1]  # [n_agent, 1 ,1]

        # half_cos = 0.5 * head.cos()  # [n_agent, n_step, n_target]
        # half_sin = 0.5 * head.sin()  # [n_agent, n_step, n_target]
        half_cos = np.cos(head) * 0.5  # [n_agent, n_step, n_target]
        half_sin = np.sin(head) * 0.5  # [n_agent, n_step, n_target]

        length_cos = length * half_cos  # [n_agent, n_step, n_target]
        length_sin = length * half_sin  # [n_agent, n_step, n_target]
        width_cos = width * half_cos  # [n_agent, n_step, n_target]
        width_sin = width * half_sin  # [n_agent, n_step, n_target]

        left_front_x = x + length_cos - width_sin
        left_front_y = y + length_sin + width_cos
        left_front = np.stack((left_front_x, left_front_y), axis=-1)

        right_front_x = x + length_cos + width_sin
        right_front_y = y + length_sin - width_cos
        right_front = np.stack((right_front_x, right_front_y), axis=-1)

        right_back_x = x - length_cos + width_sin
        right_back_y = y - length_sin - width_cos
        right_back = np.stack((right_back_x, right_back_y), axis=-1)

        left_back_x = x - length_cos - width_sin
        left_back_y = y - length_sin + width_cos
        left_back = np.stack((left_back_x, left_back_y), axis=-1)

        polygon_contour = np.stack(
            (left_front, right_front, right_back, left_back), axis=-2
        )

        return polygon_contour

    def get_nearest_traj(self, x, y, grid_mask, trajs_in_bin):
        valid_pos = np.argwhere(grid_mask)
        distances = np.abs(valid_pos[:, 0] - x) + np.abs(valid_pos[:, 1] - y)
        nearest_idx = np.argmin(distances)
        nearest_x, nearest_y = valid_pos[nearest_idx]
        nearest_traj = np.array(trajs_in_bin[nearest_x][nearest_y]).mean(axis=0)
        return nearest_traj    

    def selection_indices(total, requested):
        if requested is None or requested >= total:
            return None
        return np.linspace(0, total - 1, requested, dtype=np.int64)

    def aggregate_class(self, trajectories, agent_class):
        x_min, x_max = self.x_min[agent_class], self.x_max[agent_class]
        y_min, y_max = self.y_min[agent_class], self.y_max[agent_class]
        x_binnum = self.x_binnum[agent_class]
        y_binnum = self.y_binnum[agent_class]
        num_cells = x_binnum * y_binnum

        indices = self.selection_indices(
            len(trajectories), self.max_traj_nums[agent_class]
        )
        selected = len(trajectories) if indices is None else len(indices)
        grid_count = np.zeros(num_cells, dtype=np.int64)
        grid_sum = np.zeros(
            (num_cells, self.shift + 1, 3), dtype=np.float64
        )
        flip_sign = np.array([1.0, -1.0, -1.0], dtype=np.float64)

        for start in range(0, selected, self.aggregation_chunk_size):
            stop = min(start + self.aggregation_chunk_size, selected)
            if indices is None:
                chunk = np.asarray(trajectories[start:stop])
            else:
                chunk = np.asarray(trajectories[indices[start:stop]])

            chunk_yaw = (
                chunk[:, :, 2].astype(np.float64, copy=False) + np.pi
            ) % (2.0 * np.pi) - np.pi
            endpoint_x = chunk[:, -1, 0]
            endpoint_y = chunk[:, -1, 1]
            grid_x = np.round(
                (endpoint_x - x_min) / (x_max - x_min) * x_binnum
            ).astype(np.int32)
            grid_y = np.round(
                (endpoint_y - y_min) / (y_max - y_min) * y_binnum
            ).astype(np.int32)
            grid_y_flip = np.round(
                (-endpoint_y - y_min) / (y_max - y_min) * y_binnum
            ).astype(np.int32)

            path_valid = (
                np.abs(chunk[:, :, 0]).sum(axis=1) / (self.shift + 1) < x_max
            ) & (
                np.abs(chunk[:, :, 1]).sum(axis=1) / (self.shift + 1) < y_max
            )
            valid = (
                path_valid
                & (grid_x >= 0)
                & (grid_x < x_binnum)
                & (grid_y >= 0)
                & (grid_y < y_binnum)
            )
            valid_flip = (
                path_valid
                & (grid_x >= 0)
                & (grid_x < x_binnum)
                & (grid_y_flip >= 0)
                & (grid_y_flip < y_binnum)
            )
            cell = grid_x[valid].astype(np.int64) * y_binnum + grid_y[valid]
            cell_flip = (
                grid_x[valid_flip].astype(np.int64) * y_binnum
                + grid_y_flip[valid_flip]
            )

            grid_count += np.bincount(cell, minlength=num_cells)
            if self.flip_trajs:
                grid_count += np.bincount(cell_flip, minlength=num_cells)

            for time_index in range(self.shift):
                values = chunk[:, time_index, :].astype(np.float64, copy=True)
                values[:, 2] = chunk_yaw[:, time_index]
                for feature_index in range(3):
                    grid_sum[:, time_index + 1, feature_index] += np.bincount(
                        cell,
                        weights=values[valid, feature_index],
                        minlength=num_cells,
                    )
                    if self.flip_trajs:
                        grid_sum[:, time_index + 1, feature_index] += np.bincount(
                            cell_flip,
                            weights=(
                                values[valid_flip, feature_index]
                                * flip_sign[feature_index]
                            ),
                            minlength=num_cells,
                        )

            print(
                f"{agent_class}: aggregated {stop:,}/{selected:,} "
                f"({100.0 * stop / selected:5.1f}%)",
                flush=True,
            )

        grid_mean = np.zeros_like(grid_sum)
        occupied = grid_count > 0
        grid_mean[occupied] = (
            grid_sum[occupied] / grid_count[occupied, None, None]
        )
        return (
            grid_count.reshape(x_binnum, y_binnum),
            grid_mean.reshape(
                x_binnum, y_binnum, self.shift + 1, 3
            ),
        )

    def _interpolate_curves(self, end_x, end_y, random_values):
        a = random_values[:, 0] * 4.0 - 2.0
        b = random_values[:, 1] * 4.0 - 2.0
        a_y = random_values[:, 2] * 4.0 - 2.0
        t = self.t
        c = (end_x - a * t**3 - b * t**2) / t
        b_y = (end_y - a_y * t**3) / t**2
        time = np.arange(self.shift + 1, dtype=np.float64) * 0.1
        traj_x = (
            a[:, None] * time[None, :] ** 3
            + b[:, None] * time[None, :] ** 2
            + c[:, None] * time[None, :]
        )
        traj_y = (
            a_y[:, None] * time[None, :] ** 3
            + b_y[:, None] * time[None, :] ** 2
        )
        position = np.stack((traj_x, traj_y), axis=-1)
        heading = np.arctan2(np.diff(traj_y, axis=1), np.diff(traj_x, axis=1))
        heading = np.concatenate(
            (np.zeros((len(traj_x), 1)), heading), axis=1
        )
        return np.concatenate((position, heading[:, :, None]), axis=-1)

    def interpolate_curve(self, x, y):
        return self._interpolate_curves(
            np.asarray([x]),
            np.asarray([y]),
            self.rng.rand(1, 3),
        )[0]

    
    def get_trajtok_vocab(self):
        self.rng = np.random.RandomState(self.random_seed)
        self.vocab = {}
        self.vocab['token'] = {}
        self.vocab['traj'] = {}
        self.vocab['token_all'] = {}
        self.vocab['grid_mask'] = {}
        self.vocab['grid_mask_filtered'] = {}        
        self.vocab['raw_ep'] = {}
        
        for agent_class in self.agent_classes:

            x_binnum, y_binnum = self.x_binnum[agent_class], self.y_binnum[agent_class]
            x_min, x_max = self.x_min[agent_class], self.x_max[agent_class]
            y_min, y_max = self.y_min[agent_class], self.y_max[agent_class]
            filter_range = self.filter_range[agent_class]
            filter_threshold_add = self.filter_threshold_add[agent_class]
            filter_threshold_remove = self.filter_threshold_remove[agent_class]
            valid_count_threshold = self.valid_count_threshold[agent_class]

            grid_mask_count, grid_mean = self.aggregate_class(
                self.traj_data[agent_class], agent_class
            )
            raw_cells = np.indices((x_binnum, y_binnum)).reshape(2, -1).T
            raw_eps = np.column_stack((
                raw_cells[:, 0] * (x_max - x_min) / x_binnum + x_min,
                raw_cells[:, 1] * (y_max - y_min) / y_binnum + y_min,
            ))
            self.vocab['raw_ep'][agent_class] = raw_eps
            grid_mask = (grid_mask_count >= valid_count_threshold)

            grid_mask_filtered = grid_mask.copy()
            for x in range(x_binnum):
                for y in range(y_binnum):
                    neighbors = grid_mask[max(0,x-filter_range):min(x_binnum,x+filter_range+1),max(0,y-filter_range):min(y_binnum,y+filter_range+1)]
                    if grid_mask[x,y] and neighbors.sum() < filter_threshold_remove:
                        grid_mask_filtered[x,y] = False
                    if not grid_mask[x,y] and neighbors.sum() > filter_threshold_add:
                        grid_mask_filtered[x,y] = True          
            cells = np.argwhere(grid_mask_filtered)
            empirical = grid_mask[cells[:, 0], cells[:, 1]]
            endpoint_xy = raw_eps[cells[:, 0] * y_binnum + cells[:, 1]]
            token_trajs = np.zeros(
                (len(cells), self.shift + 1, 3), dtype=np.float64
            )
            token_trajs[empirical] = grid_mean[
                cells[empirical, 0], cells[empirical, 1]
            ]
            token_trajs[empirical, -1, :2] = endpoint_xy[empirical]

            generated = ~empirical
            token_trajs[generated] = self._interpolate_curves(
                endpoint_xy[generated, 0],
                endpoint_xy[generated, 1],
                self.rng.rand(int(generated.sum()), 3),
            )
            if agent_class == "veh":
                width_length = np.array([2.0, 4.8])
            elif agent_class == "ped":
                width_length = np.array([1.0, 1.0])
            elif agent_class == "cyc":
                width_length = np.array([1.0, 2.0])
            token_countour = self.cal_polygon_contour(
                token_trajs[:, :, 0:2], token_trajs[:, :, 2], width_length=width_length
            )# [n_token, shift+1, 4, 2]
            token = token_countour[:, -1, :, :]
            self.vocab['traj'][agent_class] = token_trajs
            self.vocab['token'][agent_class] = token
            self.vocab['token_all'][agent_class] = token_countour
            self.vocab['grid_mask'][agent_class] = grid_mask
            self.vocab['grid_mask_filtered'][agent_class] = grid_mask_filtered
            print(agent_class, token_countour.shape)

        with open(self.output_path, 'wb') as f:
            pickle.dump(self.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('token vocab generated')

if __name__ == '__main__':
    generator = TrajTok()
    generator.get_trajtok_vocab()
