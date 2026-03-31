import math
from pathlib import Path

import hydra
import torch
#torch.autograd.set_detect_anomaly(True)
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import os, pickle
from src.smart.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetric,
    WOSACSubmission,
    minADE,
)
import time
import numpy as np
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts
from src.smart.utils import split_by_type, merge_by_type

class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        self.calculate_wosac_metrics = model_config.calculate_wosac_metrics if hasattr(model_config, "calculate_wosac_metrics") else False
        self.wosac_metrics_version = model_config.wosac_metrics_version if hasattr(model_config, "wosac_metrics_version") else "2025"
        self.result_save_dir = model_config.result_save_dir if hasattr(model_config, "result_save_dir") and \
            model_config.result_save_dir != 'None' else None

        self.token_processor = TokenProcessor(**model_config.token_processor)
        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=1)
        self.wosac_metric = WOSACMetric(self.wosac_metrics_version)
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.cls_loss = CrossEntropy(**model_config.training_loss)


        if self.result_save_dir is not None:
            os.makedirs(self.result_save_dir, exist_ok=True)
            
        self.inference_time = []

    def training_step(self, data, batch_idx):
        with torch.no_grad():
            tokenized_map, tokenized_agent = self.token_processor(data)
        pred = self.encoder(tokenized_map, tokenized_agent)
        loss = self.cls_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            token_heading=tokenized_agent["token_heading"],
            train_mask=data["agent"]["train_mask"],  # [n_agent]
            current_epoch=self.current_epoch,
        )

        self.log("train/loss", loss, prog_bar=True, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.cls_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
                token_heading=tokenized_agent["token_heading"],
            )
            self.log("val_open/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=1)
            type_mask = tokenized_agent["type_mask"]
            pred_by_type = pred["next_token_logits"]
            pred_valid_by_type = split_by_type(pred["next_token_valid"], type_mask)
            target_by_type = split_by_type(tokenized_agent["token_idx"], type_mask)
            target_valid_by_type = split_by_type(tokenized_agent["valid_mask"], type_mask)
            for agent_type in type_mask.keys():
                if pred_valid_by_type[agent_type].sum() > 0:
                    self.TokenCls.update(
                        pred=pred_by_type[agent_type],
                        pred_valid=pred_valid_by_type[agent_type],
                        target=target_by_type[agent_type][:, 2:],
                        target_valid=target_valid_by_type[agent_type][:, 2:],
                    )
            self.log("val_open/acc", self.TokenCls, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=1,)

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):

                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])  

            pred_traj = torch.stack(pred_traj, dim=0)  # [n_rollout, n_ag, n_step, 2]
            pred_z = torch.stack(pred_z, dim=0)  # [n_rollout, n_ag, n_step]
            pred_head = torch.stack(pred_head, dim=0)  # [n_rollout, n_ag, n_step]
            simulated_states = torch.cat([pred_traj, pred_z.unsqueeze(-1), pred_head.unsqueeze(-1)], dim=-1)  # [n_rollout, n_ag, n_step, 4]
            if self.result_save_dir is not None:
                ptr = data['agent']['ptr']
                for i in range(len(ptr)-1):
                    result = {
                              'simulated_states': simulated_states[:, ptr[i]:ptr[i+1],:,:].detach().cpu().numpy(),
                              'agent_id': data['agent']['id'][ptr[i]:ptr[i+1]].detach().cpu().numpy(),
                            }
                    with open(os.path.join(self.result_save_dir,data.scenario_id[i]+'.pkl'),'wb') as f: 
                        pickle.dump(result,f)
            if self.calculate_wosac_metrics:
                self.wosac_metric.update(
                    scenario_id=data["scenario_id"],
                    gt_scenarios=data["gt_scenario"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    simulated_states=simulated_states,
                )


    def on_validation_epoch_end(self):
        if self.calculate_wosac_metrics:
            metric_dict = self.wosac_metric.compute()
            print(metric_dict)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred_traj, pred_z, pred_head, distances_to_objects_all, distances_to_road_edge_all = [], [], [], [], []
        for _ in range(self.n_rollout_closed_val):

            pred = self.encoder.inference(
                tokenized_map, tokenized_agent, self.validation_rollout_sampling
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])  

        pred_traj = torch.stack(pred_traj, dim=0)  # [n_rollout, n_ag, n_step, 2]
        pred_z = torch.stack(pred_z, dim=0)  # [n_rollout, n_ag, n_step]
        pred_head = torch.stack(pred_head, dim=0)  # [n_rollout, n_ag, n_step]

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
