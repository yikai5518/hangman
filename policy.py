from typing import Callable

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class HangmanMLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim_pi: int,
        output_dim_vf: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        self.latent_dim_pi = output_dim_pi
        self.latent_dim_vf = output_dim_vf
        
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_pi),
            nn.ReLU(),
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_vf),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)
    
    def forward_critic(self, x):
        return self.value_net(x)
    
    def forward_actor(self, x):
        return self.policy_net(x)


class HangmanPolicy(ActorCriticPolicy):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        output_dim_pi: int,
        output_dim_vf: int,
        hidden_dim: int,
        *args,
        **kwargs,
    ):
        self.output_dim_pi = output_dim_pi
        self.output_dim_vf = output_dim_vf
        self.hidden_dim = hidden_dim
        
        super().__init__(
            obs_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HangmanMLPNetwork(
            self.features_dim,
            self.output_dim_pi,
            self.output_dim_vf,
            self.hidden_dim,
        )