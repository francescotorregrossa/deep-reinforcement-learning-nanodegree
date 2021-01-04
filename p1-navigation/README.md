# Navigation
**Project 1 of Deep Reinforcement Learning Nanodegree**

![](imgs/gif_2.gif)

> The model used to generate this gif is `final.pth` (Dueling Double DQN), which was trained for 700 episodes using `main.py`.

## Overview

The environment for this project is [Banana](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) from Unity and it is provided in the `setup` folder. This repository contains an implementation of the original [DQN algorithm](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) (although not directly from pixels) and two variants, [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf) and [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf). 

For details on the implementation and comparison between the models see the [report](Report.ipynb). Alternatively, you can find some pre-trained models under `models/` and the source code in `main.py` and `code/`.

## Environment

The agent is placed in a 3D room filled with yellow and blue bananas. The goal is to **pick up as many yellow bananas as possible while avoiding the blue ones**.

The state space has **37 dimensions** and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

At each timestep, the agent can take one of **four actions**:

- `0`, move forward
- `1`, move backward
- `2`, turn left
- `3`, turn right

The reward function gives **+1 and -1** for picking up yellow and blue bananas, respectively. If no banana is picked up, the reward is zero.

The task is **episodic** and is considered solved when the agent gets an **average score of +13** over 100 consecutive episodes.

## Getting started

> Note that this was tested in macOS only

### Requirements

You'll need [conda](https://docs.conda.io/en/latest/) to prepare the environment and execute the code. 

Other resources are already available in this repository under `setup/`, so you can simply clone it.

```bash
git clone https://github.com/francescotorregrossa/deep-reinforcement-learning-nanodegree.git
cd deep-reinforcement-learning-nanodegree/p1-navigation
```

Optionally, you can install [jupyter](https://jupyter.org) if you want to work on the report notebook.

### Create a conda environment

This will create an environment named `p1_navigation` and install the required libraries.

```bash
conda create --name p1_navigation python=3.6
conda activate p1_navigation
unzip setup.zip
pip install ./setup
```

### Watch a pre-trained agent

You can use `main.py` to watch an agent play the game. The provided model `final.pth` is a **Dueling Double DQN with uniform replay buffer**.

```bash
python main.py
```

If you want to try another configuration, you can use one of the files under `model/` but note that you might also need to [change this line](https://github.com/francescotorregrossa/deep-reinforcement-learning-nanodegree/blob/8932c0b02bab125234bd3484c723549c89395b3b/p1-navigation/main.py#L54-L55) in `main.py`.

### Train an agent from scratch

You can also use `main.py` to train a new agent. Again, if you want to change the configuration you have to update [this line](https://github.com/francescotorregrossa/deep-reinforcement-learning-nanodegree/blob/8932c0b02bab125234bd3484c723549c89395b3b/p1-navigation/main.py#L54-L55). You'll find other classes and functions in the `code/` folder. The report also contains useful functions for plotting results with `matplotlib`.

```bash
python main.py -t
```

Note that this script will **override** `final.pth`.

### Open the report with jupyter

```bash
python -m ipykernel install --user --name p1_navigation --display-name "p1_navigation"
jupyter notebook
```

Make sure to set the **kernel** to `p1_notebook` after you open the report.

### Uninstall

```bash
conda deactivate
conda remove --name p1_navigation --all
jupyter kernelspec uninstall p1_navigation
```
