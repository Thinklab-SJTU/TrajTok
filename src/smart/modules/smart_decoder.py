from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: Dict[str, int],
    ) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder(tokenized_agent, map_feature)
        return pred_dict

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, sampling_scheme
        )
        return pred_dict
