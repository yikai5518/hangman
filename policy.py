from typing import Callable

import torch
from torch import nn
from torch.distributions import Categorical
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution


class HangmanFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        obs_space: spaces.MultiDiscrete,
        embedding_size: int,
    ):
        super().__init__(obs_space, embedding_size * (obs_space.shape[0] - 26))
        
        self.obs_space = obs_space
        self.emb = nn.Embedding(26 + 2, embedding_size)
    
    def forward(self, observation):
        offsets = torch.zeros((self._observation_space.nvec.size + 1,))
        offsets[1:] = torch.cumsum(torch.tensor(self._observation_space.nvec.flatten()), dim=0)
        indices = torch.nonzero(observation)
        offsets = offsets[:-1].repeat(indices.shape[0] // (offsets.shape[0] - 1))
        indices = indices[:, 1] - offsets
        x = indices.reshape((-1, self._observation_space.shape[0])).long()

        states = x[:, :-26]
        mask = x[:, -26:]
        
        states = self.emb(states)
        return torch.concat((states.flatten(start_dim=1), mask), dim=1)


class FixedCategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        return action_logits

    def proba_distribution(self, action_logits):
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions):
        return self.distribution.log_prob(actions)

    def entropy(self):
        return self.distribution.entropy()

    def sample(self):
        return self.distribution.sample()

    def mode(self):
        return torch.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits, deterministic: bool = False):
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits):
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class HangmanMLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim_vf: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        self.latent_dim_pi = 26
        self.latent_dim_vf = output_dim_vf
                
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 26),
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
        return self.value_net(x[:, :-26])
    
    def forward_actor(self, x):
        states = x[:, :-26]
        mask = x[:, -26:]
        
        x = self.policy_net(states)
        x = x * mask
        x = x + -1e5 * (x == 0)
        
        return x


class HangmanPolicy(ActorCriticPolicy):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        output_dim_vf: int,
        hidden_dim: int,
        *args,
        **kwargs,
    ):
        self.output_dim_vf = output_dim_vf
        self.hidden_dim = hidden_dim
        
        super().__init__(
            obs_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
        self.action_dist = FixedCategoricalDistribution(action_space.n)
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HangmanMLPNetwork(
            self.features_dim,
            self.output_dim_vf,
            self.hidden_dim,
        )
        
    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(latent_pi)