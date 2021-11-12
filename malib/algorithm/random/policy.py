# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import torch
import gym

from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType
from malib.algorithm.common.model import get_model


class RandomPolicy(Policy):
    def __init__(
        self,
        registered_name,
        observation_space,
        action_space,
        model_config,
        custom_config,
    ):
        super().__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.set_actor(
            get_model(self.model_config["actor"])(observation_space, action_space)
        )
        self.set_critic(
            get_model(self.model_config["critic"])(
                observation_space, gym.spaces.Discrete(1)
            )
        )

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        raise NotImplementedError

    def compute_action(self, observation: DataTransferType, **kwargs) -> Any:
        logits = torch.softmax(self.actor(observation), dim=-1)
        action_prob = torch.zeros(self.action_space.n)
        if "legal_moves" in kwargs:
            mask = torch.zeros_like(logits)
            mask[kwargs["legal_moves"]] = 1
        elif "action_mask" in kwargs:
            mask = torch.FloatTensor(kwargs["action_mask"])
        else:
            mask = torch.ones_like(logits)
        logits = mask * logits
        action = logits.argmax(dim=-1).view((-1, 1)).squeeze(-1).numpy()
        return action, None, {"action_probs": action_prob}

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {}

    def set_weights(self, parameters):
        pass
