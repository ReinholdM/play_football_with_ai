''' 
put this file in the directory of gfootball :football/gfootball/env/players/

and run :

python3 -m gfootball.play_game --players "keyboard:left_players=1;bot:left_player=1;football_ai:right_players=1, \
    checkpoint=$PATH" --action_set=full

$PATH is the path where model are loaded;
--players means there is one left player controlled by keyboard and one controlled by bulit-in AI, one right player \
    controlled by our own model.
'''

from malib.algorithm.common.misc import hard_update
import os
import gym
import pickle
import numpy as np
import torch
from malib.algorithm.mappo import MAPPO
from malib.envs.gr_football.encoders import encoder_basic
from gfootball.env import football_action_set
from gfootball.env import player_base


class Player(player_base.PlayerBase):
  """An agent loaded from torch model."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)

    self._action_set = (env_config['action_set']
                        if 'action_set' in env_config else 'default')
    self._actor = load_model(player_config['checkpoint'])

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'

    feature_encoder = encoder_basic.FeatureEncoder()
    observation = feature_encoder.encode(observation[0])
    observation = concate_observation_from_raw(observation)
    #print("-"*20, observation)
    logits = self._actor(observation)
    illegal_action_mask = torch.FloatTensor(
            1 - observation[..., : logits.shape[-1]]
        ).to(logits.device)
    assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
            illegal_action_mask.max(),
            illegal_action_mask.min(),
        )
    logits = logits - 1e10 * illegal_action_mask
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample().numpy()
    actions = [football_action_set.action_set_dict[self._action_set][action]]
    return actions

def load_model(load_path):
    with open(os.path.join(load_path, "desc.pkl"), "rb") as f:
        desc_pkl = pickle.load(f)
    res = MAPPO(
        desc_pkl["registered_name"],
        desc_pkl["observation_space"],
        desc_pkl["action_space"],
        desc_pkl["model_config"],
        desc_pkl["custom_config"]
    )
    actor = torch.load(os.path.join(load_path, "actor.pt"), res.device)
    hard_update(res._actor, actor)
    return actor

def concate_observation_from_raw(obs):
    obs_cat = np.hstack(
        [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
    )
    return obs_cat

