# -*- coding: utf-8 -*-
import copy
from malib.algorithm.mappo.utils import init_fc_weights
import os
import pickle
from typing import Tuple, Any, Dict
import gym
import torch
from torch import nn
from malib.algorithm.common.model import get_model
from malib.algorithm.common.policy import Policy
from malib.utils.typing import DataTransferType
from malib.algorithm.common.misc import hard_update


class MAPPO(Policy):
    def __init__(
            self,
            registered_name: str,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            model_config: Dict[str, Any] = None,
            custom_config: Dict[str, Any] = None,
    ):
        super(MAPPO, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.opt_cnt = 0

        # Todo (Linghui): hypothesize we had only one reward factor
        self.reward_weights_num = 1

        self.register_state(self.opt_cnt, "opt_cnt")

        self._use_q_head = custom_config["use_q_head"]

        actor = get_model(self.model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )

        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )

        global_observation_space = custom_config["global_state_space"]
        critic = get_model(self.model_config["critic"])(
            global_observation_space,
            action_space if self._use_q_head else gym.spaces.Discrete(1),
            custom_config.get("use_cuda", False),
        )

        meta_network = get_model(self.model_config["actor"])(
            observation_space, gym.spaces.Discrete(5), custom_config.get("use_cuda", False)
        )  # todo: the output units num = 8 which is the reward factor

        use_orthogonal = model_config["initialization"]["use_orthogonal"]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_weights(m):
            if type(m) == nn.Linear:
                init_fc_weights(m, init_method, model_config["initialization"]["gain"])

        actor.apply(init_weights)
        critic.apply(init_weights)
        meta_network.apply(init_weights)

        # register state handler
        self.set_actor(actor)
        self.set_critic(critic)
        self.set_meta(meta_network)

        self.register_state(self._actor, "actor")
        self.register_state(self._critic, "critic")
        self.register_state(self._meta, "meta")

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy._actor = self_copy._actor.to(device)
        self_copy._critic = self_copy._critic.to(device)
        self_copy._meta = self_copy._meta.to(device)
        return self_copy

    def compute_actions(self, observation, **kwargs):
        raise RuntimeError("Shouldn't use it currently")

    def compute_action(self, observation, **kwargs):
        logits = self.actor(observation)
        illegal_action_mask = torch.FloatTensor(
            1 - observation[..., : logits.shape[-1]]
        ).to(logits.device)
        assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
            illegal_action_mask.max(),
            illegal_action_mask.min(),
        )
        logits = logits - 1e10 * illegal_action_mask
        if "action_mask" in kwargs:
            raise NotImplementedError
        dist = torch.distributions.Categorical(logits=logits)
        extra_info = {}
        action_prob = dist.probs.detach().numpy()  # num_action

        # if observation is not None:
        #     for n in range(logits.shape[0]):
        #         action_prob[n][observation[n][:logits.shape[1]]==0] = 0.
        extra_info["action_probs"] = action_prob
        action = dist.sample().numpy()
        if "share_obs" in kwargs and kwargs["share_obs"] is not None:
            extra_info["value"] = self.critic(kwargs["share_obs"]).detach().numpy()
        return action, action_prob, extra_info

    def compute_reward_weights(self, observation, **kwargs):
        logits = self.meta(observation)
        dist = torch.distributions.Categorical(logits=logits)
        reward_weights = dist.probs.detach().numpy()

        return reward_weights

    def train(self):
        pass

    def eval(self):
        pass

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def dump(self, dump_dir):
        torch.save(self._actor, os.path.join(dump_dir, "actor.pt"))
        torch.save(self._critic, os.path.join(dump_dir, "critic.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)
        
        res = MAPPO(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"]
        )

        actor = torch.load(os.path.join(dump_dir, "actor.pt"), res.device)
        critic = torch.load(os.path.join(dump_dir, "critic.pt"), res.device)
        
        hard_update(res._actor, actor)
        hard_update(res._critic, critic)
        return res



if __name__ == "__main__":
    from malib.envs.gr_football import env, default_config
    import yaml
    cfg = yaml.load(open("examples/mappo_grfootball/mappo_team_psro.yaml"))
    env = env(**default_config)
    custom_cfg = cfg["algorithms"]["MAPPO"]["custom_config"]
    custom_cfg.update({"global_state_space": env.state_space})
    policy = MAPPO(
        "MAPPO",
        env.observation_spaces["team_0"],
        env.action_spaces["team_0"],
        cfg["algorithms"]["MAPPO"]["model_config"],
        custom_cfg
    )
    policy.dump("play")
    MAPPO.load("play")