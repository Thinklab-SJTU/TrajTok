import numpy as np
import os
import pickle, json
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager
from google.protobuf import text_format
from queue import Empty
from waymo_open_dataset.protos import sim_agents_metrics_pb2
import fast_sim_agents_metrics.metrics as sim_agents_metric_api
from tqdm import tqdm
from argparse import ArgumentParser
from scenario_gt_converter import gt_scenario_to_device


metric_names = ['metametric', 'average_displacement_error', 'min_average_displacement_error', 'linear_speed_likelihood', 'linear_acceleration_likelihood', 'angular_speed_likelihood',
                'angular_acceleration_likelihood', 'distance_to_nearest_object_likelihood', 'collision_indication_likelihood', 'time_to_collision_likelihood', 'distance_to_road_edge_likelihood',
                'offroad_indication_likelihood',  'simulated_collision_rate', 'simulated_offroad_rate']

def worker(device, predict_dir, gt_scenario_dir, sim_agent_eval_config, file_queue, result_queue, version):
    if 'cuda' in device:
        torch.cuda.set_device(device)
    while True:
        try:
            file = file_queue.get_nowait()
            try:
                with open(os.path.join(gt_scenario_dir, file), 'rb') as f:
                    gt_scenario = pickle.load(f)
                with open(os.path.join(predict_dir, file), 'rb') as f:
                    predict = pickle.load(f)    
                gt_scenario = gt_scenario_to_device(gt_scenario, device=device)
                predict['agent_id'] = torch.tensor(predict['agent_id'], device=device)
                predict['simulated_states'] = torch.tensor(predict['simulated_states'], device=device)
                scenario_metrics = sim_agents_metric_api.compute_scenario_metrics_for_bundle(sim_agent_eval_config, gt_scenario, predict, version)
                result_queue.put(scenario_metrics)
            except Exception as e:
                print(f"Error computing metrics {file} on {device}: {str(e)}")
        except Empty:
            break

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument(
        "--predict_dir",
        type=str,
        required=True,
        help='Path of saved rollouts.'
    )
    parser.add_argument(
        "--gt_scenario_dir", 
        type=str, 
        required=True,
        help='Path of processed GT files'
    )
    parser.add_argument(
        "--version", 
        type=str, 
        choices=["2024", "2025"],
        default="2025",
        help='version of WOSAC metrics. Support 2024 and 2025.'
        )
    parser.add_argument(
        "--metric_save_path", 
        type=str, 
        default=None,
        help='path to save result'
        )    
    parser.add_argument(
        "--num_gpus", 
        default=None,
        type=int,
        )
    parser.add_argument(
        "--debug", 
        action="store_true"
        )   
    
    args = parser.parse_args()    
    files = os.listdir(args.predict_dir) #[0:1000]
    if args.version == '2024':
        proto_path = 'fast_sim_agents_metrics/challenge_2024_config.textproto'
    elif args.version == '2025':
        proto_path = 'fast_sim_agents_metrics/challenge_2025_sim_agents_config.textproto'
    with open(proto_path,'r') as f:
        sim_agent_eval_config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), sim_agent_eval_config)
        
    num_gpus = min(args.num_gpus, torch.cuda.device_count()) if args.num_gpus else torch.cuda.device_count()
    results = []
    if args.debug or num_gpus == 0:
        device = 'cuda:0' if args.debug and num_gpus != 0 else 'cpu'
        for file in tqdm(files, desc="Computing Metrics"):
            with open(os.path.join(args.gt_scenario_dir, file), 'rb') as f:
                gt_scenario = pickle.load(f)
            with open(os.path.join(args.predict_dir, file), 'rb') as f:
                predict = pickle.load(f)  
            predict['agent_id'] = torch.tensor(predict['agent_id'], device=device)
            predict['simulated_states'] = torch.tensor(predict['simulated_states'], device=device)
            gt_scenario = gt_scenario_to_device(gt_scenario, device=device)
            scenario_metrics = sim_agents_metric_api.compute_scenario_metrics_for_bundle(sim_agent_eval_config, gt_scenario, predict, args.version)
            results.append(scenario_metrics)
    else:
        manager = Manager()
        file_queue = manager.Queue()
        result_queue = manager.Queue()
        for file in files:
            file_queue.put(file)
        processes = []
        for i in range(num_gpus):
            p = Process(target=worker, args=(f"cuda:{i}", args.predict_dir, args.gt_scenario_dir, sim_agent_eval_config, file_queue, result_queue, args.version))
            p.start()
            processes.append(p)
        with tqdm(total=len(files), desc="Computing Metrics") as pbar:
            while len(results) < len(files):
                result = result_queue.get()
                results.append(result)
                pbar.update(1)
        for p in processes:
            p.join()        

    final_result = {}
    if args.version == '2025':
        metric_names += ['traffic_light_violation_likelihood', 'simulated_traffic_light_violation_rate']
    for metric_name in metric_names:
        final_result[metric_name] = 0
        for result in results:
            final_result[metric_name] += result[metric_name]
        final_result[metric_name] /= len(results)
        
    print("evaluation results:")
    print(final_result)
    if args.metric_save_path is not None:
        with open(args.metric_save_path, 'w') as f:
            json.dump(final_result, f)
        print(f"results saved to {args.metric_save_path}")

