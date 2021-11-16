# Play with Football AI Scripts

## Installation

```shell
conda create -n malib python==3.7 -y
conda activate malib
pip install -e .
# for development
# pip install -e .[dev]
make test
# then install the requirements.txt
pip install -r requirements.txt
```

## Play With football AI Trained by Malib

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

1. open this repo in your terminal, `cd play_football_with_human`

2. `bash play_with_human.sh $PATH`

   $PATH is the path where your <gfootball environment>  are installed such as `~/miniconda3/env/env_name/lib/python3.6/site-packages/gfootball/`.

3. extract trajectories from the directory that has stored dump files by running `python mappo_grfootball/dump_to_trjectories.py`
