#/bin/bash!
local_dir=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${local_dir}"
export PYTHONPATH="${PYTHONPATH}:${local_dir}/malib"

cp mappo_grfootball/football_ai.py $1/env/players/
cp mappo_grfootball/malib_5_vs_5.py $1/scenarios/

ai_path=${local_dir}/run_and_goal/

python3 -m gfootball.play_game --players "keyboard:left_players=1;football_ai:right_players=1,checkpoint=${ai_path};football_ai:right_players=1,checkpoint=${ai_path};football_ai:right_players=1,checkpoint=${ai_path};football_ai:right_players=1,checkpoint=${ai_path}" --action_set=full --level "malib_5_vs_5"
