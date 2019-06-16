# Introduction
It's a research about reinforcement learning.
I use [pysc2](https://github.com/deepmind/pysc2) as environment.The original paper is [sc2le](https://deepmind.com/documents/110/sc2le.pdf), you can see and know how it work, and algorithm is [a3c](https://arxiv.org/abs/1602.01783).

# Requirement
1. of course. the StarCraft II game, and remember to download the [mini game](https://github.com/Blizzard/s2client-proto)
2. pysc2 (version is 2.0.1)
3. numpy
4. tensorflow-gpu (version is 1.10)
5. absl-py

# How to run
Because my system is windows 10 and anaconda of python3.7, so I create a python environment named "tf-gpu" and use python3.5 to run, you can skip the first and the last step if you are using another system.

1. `activate tf-gpu`
2. `cd my-pysc2`
3. `python main.py`
4. `deactivate`

### testing
`python main.py --map=MoveToBeacon --train=False --envs=1 --exploration_mode=eg`

# Reference
The code is based on [here](https://github.com/xhujoy/pysc2-agents). I really appreciat.

# Apologize
I'm not familiar with tensorflow and numpy, and just learning reinforcement learning. So the code has many annotation.