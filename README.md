# malib
A general-purpose multi-agent training framework.

## Installation

```shell
conda create -n malib python==3.7 -y
conda activate malib
pip install -e .
# for development
# pip install -e .[dev]
make test
# then you can run training examples
```

## Quick Start

```python
from malib.envs.poker import poker_aec_env as leduc_holdem
from malib.runner import run
from malib.rollout import rollout_func


env = leduc_holdem.env(fixed_player=True)
run(
    agent_mapping_func=lambda agent_id: agent_id,
    env_description={
        "creator": leduc_holdem.env,
        "config": {"fixed_player": True},
        "id": "leduc_holdem",
        "possible_agents": env.possible_agents,
    },
    training={
        "interface": {
            "type": "independent",
            "observation_spaces": env.observation_spaces,
            "action_spaces": env.action_spaces
        },
    },
    algorithms={
        "PSRO_PPO": {
            "name": "PPO",
            "custom_config": {
                "gamma": 1.0,
                "eps_min": 0,
                "eps_max": 1.0,
                "eps_decay": 100,
            },
        }
    },
    rollout={
        "type": "async",
        "stopper": "simple_rollout",
        "callback": rollout_func.psro
    }
)

### Play With football AI Trained by Malib

#### 1. Environment 

Windows10 2004 or higher 

Ubuntu20.04 

#### 2. Installation

1. Install wsl2 by running Command Prompt as administrator, and input

`wsl --list --online` 

2. Type following command  to install the WSL with a specific distro on Win10 and enter:

`wsl --install -d Ubuntu-20.04`

Alternative of steps 2 :  find Ubuntu-20.04 distro in  Windows store, and install it

3. Restart your winPC

4. install gfootball on Ubuntu terminal (ensure pip and apt work properly), refers to ：

   https://github.com/google-research/football#on-your-computer

5. install malib(necessary for now) 
   install dependency: pip install -r requirements.txt
   add malib_dev/, malib_dev/malib to environment variable PYTHONPATH
   then you can import malib from python command

6. Install xrdp to enable GUI interface of WSL, refers to :

   https://zhuanlan.zhihu.com/p/149501381

#### 3. Playing the game

1. copy mappo_grfootball/football_ai.py to gfootball installation directory  gfootball/env/players/.

2. copy mappo_grfootball/malib_5_vs_5.py to gfootball/scenarios which provides specific football game setting 

3. Connect to Ubuntu GUI interface through RDP on win10

4. Start football game with screen rendering with flags：

   5 vs 5 full game: 

   left_team: keyboard, 3 built-AI teammate and 1 build-in GK; right_team: 4 opponents controlled by your trained model and 1 built-in GK;

   `python3 -m gfootball.play_game --player "keyboard:left_players=1;football_ai:right_players=1,checkpoint=$PATH;football_ai:right_players=1,checkpoint=$PATH;football_ai:right_players=1,checkpoint=$PATH;football_ai:right_players=1, checkpoint=$PATH" --action_set=full --level "malib_5_vs_5"`

   $PATH is the path where your model are saved.

   
```
