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
        self.y_max = {'veh': 3, 'ped': 2, 'cyc': 1}
        self.y_min = {'veh': -2, 'ped': -2, 'cyc': -1}
        self.x_binnum = {'veh': 250, 'ped': 120, 'cyc': 180} # 0.05
        self.y_binnum = {'veh': 80, 'ped': 80, 'cyc': 40}  # 0.05
        # filter settings
        self.valid_count_threshold = {'veh': 3, 'ped': 1, 'cyc': 1}
        self.filter_range = {'veh': 4, 'ped': 4, 'cyc': 4}
        self.filter_threshold_add = {'veh': 20, 'ped': 20, 'cyc': 20}
        self.filter_threshold_remove = {'veh': 20, 'ped': 20, 'cyc': 20}
        # logged data extracting settings
        self.raw_data_path = 'data/waymo_processed_catk/training'
        self.traj_data_path = 'data/waymo_processed_catk/traj_data.pkl'
        self.max_workers = 16
        self.max_file_nums = 50000
        self.max_traj_nums = 12000000
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

    def interpolate_curve(self, x, y, theta, weight_factor0=0, weight_factor1=0, num_points=6):

        p0 = np.array([0, 0])
        p1 = np.array([x, y])
        dist = np.linalg.norm(p1 - p0)
        t0 = np.array([1, 0]) * dist * weight_factor0
        t1 = np.array([np.cos(theta), np.sin(theta)]) * dist * weight_factor1
        t_vals = [0, 1]                    
        points = np.vstack((p0, p1))       
        tangents = np.vstack((t0, t1))      
        spline = CubicHermiteSpline(t_vals, points, tangents)
        t_curve = np.linspace(0, 1, num_points)
        derivatives = spline(t_curve, nu=1)
        xys = spline(t_curve)
        headings = np.arctan2(derivatives[:, 1], derivatives[:, 0])
        curve_points = np.concatenate([xys, headings[:, None]], axis=-1)
        
        return curve_points   

    
    def get_trajtok_vocab(self):
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

            grid_mask_count = np.zeros((x_binnum, y_binnum))
            traj_in_bin = [[[] for _ in range(y_binnum)] for _ in range(x_binnum)]
            trajs = np.concatenate([np.zeros((self.traj_data[agent_class].shape[0],1,3)),
                                        self.traj_data[agent_class]], axis=1) #.numpy()
            if self.max_traj_nums:
                trajs = trajs[:self.max_traj_nums]

            if self.flip_trajs:
                flip = trajs.copy()
                flip[:,:,1] = -flip[:,:,1]
                flip[:,:,2] = -flip[:,:,2]
                trajs = np.concatenate([trajs, flip], axis=0)

            grid_end_x = np.round((trajs[:, self.shift, 0] - x_min) /
                                     (x_max - x_min) * x_binnum).astype(np.int32)
            grid_end_y = np.round((trajs[:, self.shift, 1] - y_min) /
                                     (y_max - y_min) * y_binnum).astype(np.int32)
            mask = (grid_end_x >= 0) & (grid_end_x < x_binnum) & \
                    (grid_end_y >= 0) & (grid_end_y < y_binnum) & \
                    (np.abs(trajs[:, :, 0]).mean(axis=-1) < x_max) & \
                    (np.abs(trajs[:, :, 1]).mean(axis=-1) < y_max)
            
            grid_end_x = grid_end_x[mask]
            grid_end_y = grid_end_y[mask]
            trajs = trajs[mask]

            for i in range(len(trajs)):
                traj_in_bin[grid_end_x[i]][grid_end_y[i]].append(trajs[i])

            raw_eps = []
            for x in range(x_binnum):
                for y in range(y_binnum):
                    grid_mask_count[x][y] = len(traj_in_bin[x][y])
                    raw_eps.append([x * (x_max - x_min) / x_binnum + x_min,
                                    y * (y_max - y_min) / y_binnum + y_min])
            self.vocab['raw_ep'][agent_class] = np.array(raw_eps)
            grid_mask = (grid_mask_count >= valid_count_threshold)

            grid_mask_filtered = grid_mask.copy()
            for x in range(x_binnum):
                for y in range(y_binnum):
                    neighbors = grid_mask[max(0,x-filter_range):min(x_binnum,x+filter_range+1),max(0,y-filter_range):min(y_binnum,y+filter_range+1)]
                    if grid_mask[x,y] and neighbors.sum() < filter_threshold_remove:
                        grid_mask_filtered[x,y] = False
                    if not grid_mask[x,y] and neighbors.sum() > filter_threshold_add:
                        grid_mask_filtered[x,y] = True          
            token_trajs = []
            for x in range(x_binnum):
                for y in range(y_binnum):
                    if not grid_mask_filtered[x,y]:
                        continue
                    
                    grid_end_x = x * (x_max - x_min) / x_binnum + x_min
                    grid_end_y = y * (y_max - y_min) / y_binnum + y_min

                    if grid_mask[x,y]:
                        token_traj = np.array(traj_in_bin[x][y]).mean(axis=0)                  
                        token_traj[-1,0] = grid_end_x
                        token_traj[-1,1] = grid_end_y
                        yaws = token_traj[:,-1]
                        if np.abs((yaws[1:] - yaws[:-1])).max() > 10 * np.pi/180:
                            nearest_traj = self.get_nearest_traj(x, y, grid_mask & grid_mask_filtered, traj_in_bin)
                            token_traj = self.interpolate_curve(grid_end_x, grid_end_y, nearest_traj[-1,2])
                    else:
                        nearest_traj = self.get_nearest_traj(x, y, grid_mask & grid_mask_filtered, traj_in_bin)
                        token_traj = self.interpolate_curve(grid_end_x, grid_end_y, nearest_traj[-1,2])
                    token_trajs.append(token_traj)

            token_trajs = np.stack(token_trajs) # [n_token, shift+1, 3]
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
            pickle.dump(self.vocab, f)
        print('token vocab generated')

generator = TrajTok()
generator.get_trajtok_vocab()