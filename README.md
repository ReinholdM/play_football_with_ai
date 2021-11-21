# Play with Football AI Scripts :notebook_with_decorative_cover:

### Environment 

Windows10 2004 or higher 

Ubuntu20.04 

### Installation

1. Install wsl2 by running Command Prompt as administrator, and input

   `wsl --list --online` 

2. Type following command  to install the WSL with a specific distro on Win10 and enter:

   `wsl --install -d Ubuntu-20.04`

Alternative of steps 2 :  find Ubuntu-20.04 distribution in  Windows store, and install it

3. Restart your winPC

4. Install conda/miniconda, then create environment with conda

   ```shell
   conda create -n malib python==3.7 -y
   conda activate malib
   
   git clone https://github.com/ReinholdM/play_football_with_ai.git
   cd play_football_with_ai
   ```
   
5. Install gfootball on Ubuntu terminal (ensure pip and apt work properly), refers to ：

   https://github.com/google-research/football#on-your-computer

6. Install malib(necessary for now) 
   install dependency: `pip install -r requirements.txt`
   then you can import malib from python command

7. Install xrdp to enable GUI interface of WSL, refers to :

   https://zhuanlan.zhihu.com/p/149501381

### Play the game!:satisfied:

1). open this repo in your terminal, `cd play_football_with_ai`

2). `bash play_with_human.sh $PATH`

   $PATH is the path where your <gfootball environment>  are installed such as `~/miniconda3/env/env_name/lib/python3.6/site-packages/gfootball/`.

3). extract trajectories from the directory that has stored dump files by running `python mappo_grfootball/dump_to_trjectories.py`  (no need)

## How to play the Football with your keyboard? :soccer:  

Make sure to check out the [keyboard mappings](https://github.com/google-research/football#keyboard-mappings). To quit the game press Ctrl+C in the terminal.
   
The replay data is stored in `/tmp/dumps/` directory. 
