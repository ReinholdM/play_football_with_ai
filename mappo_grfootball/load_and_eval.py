from malib.utils.logger import Log
from malib.algorithm.mappo import MAPPO
from malib.utils.metrics import GFootballMetric
from examples.mappo_grfootball.rollout_function import grf_simultaneous
from malib.envs.gr_football import env as create_env, default_config
from malib.envs.gr_football.vec_wrapper import DummyVecEnv, SubprocVecEnv
from argparse import ArgumentParser
import yaml
from pathlib import Path
import os
import functools
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
  
import numpy as np



parser = ArgumentParser(description="LOAD POLICY AND DO EVALUATIONS")
parser.add_argument("--policy_dir", type=str, action="append")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--num_episode", type=int, default=1)
parser.add_argument("--dump_replay", action="store_true", default=False)
parser.add_argument("--replay_dir", type=str)
parser.add_argument("--seed", type=int, default=None)

def main():
  config = parser.parse_args()
  experiment_cfg = yaml.load(open(Path(__file__).parent / "mappo_team_psro.yaml", "rb"), Loader)

  np.random.seed(config.seed)
  if config.benchmark:
    assert len(config.policy_dir)==1, \
      f"when use benchmark, number of policy_dir should be 1(got {len(config.policy_dir)})"
    env_desc = experiment_cfg["benchmark_env_description"]
    agent_interfaces = {"team_0": MAPPO.load(config.policy_dir[0])}
    # env_desc["config"]["scenario_config"]["env_name"] = "5_vs_5_hard"
  else:
    assert len(config.policy_dir)==2, \
      f"when use benchmark, number of policy_dir should be 2(got {len(config.policy_dir)})"
    env_desc = experiment_cfg["env_description"]
    agent_interfaces = {
      "team_0": MAPPO.load(config.policy_dir[0]),
      "team_1": MAPPO.load(config.policy_dir[1])
    }
  
  if config.dump_replay:
    env_desc["config"]["scenario_config"]["write_full_episode_dumps"] = True
    env_desc["config"]["scenario_config"]["logdir"] = config.replay_dir
  env_desc["creator"] = create_env
  

  env_fn = functools.partial(create_env, **env_desc["config"])

  behavior_policy_mapping = {"team_0": "team_0", "team_1": "team_1"}

  episode_seg = max(1, os.cpu_count() - 4)
  total_steps = config.num_episode // episode_seg
  episodes = [episode_seg] * total_steps + [config.num_episode - episode_seg * total_steps]

  
  with Log.stat_feedback(
    log=True,
    logger=None,
    worker_idx=None
  ) as (statisc_seq, processed_statics):
    seed = np.random.randint(0, 65536)
    print("using seed:", seed)
    for num_ep in episodes:
      if num_ep == 0:
        continue
      if "env" not in env_desc or env_desc["env"].num_envs != num_ep:
        env_desc["env"] = SubprocVecEnv([env_fn] * num_ep)
        env_desc["last_observation"] = env_desc["env"].seed(seed)
      stats, _ = grf_simultaneous(
        trainable_pairs=None,
        agent_interfaces=agent_interfaces,
        env_desc=env_desc,
        metric_type="grf",
        max_iter=-1,
        behavior_policy_mapping=behavior_policy_mapping,
        role="simulation"
      )
      seed += 1024
      
      statisc_seq.append(stats)
  total_stats = processed_statics[0]
  print("#"*40)
  print("FINISH EVALUATION,\tresults:\n")
  print(total_stats)
  print("\n", "#" * 40, "\n")



if __name__ == "__main__":
  main()