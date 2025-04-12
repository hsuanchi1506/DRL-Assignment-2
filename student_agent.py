# Remember to adjust your student ID in meta.xml
import numpy as np
import numpy
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import gdown
from collections import defaultdict
import json

from utils import TD_MCTS, TD_MCTS_Node, NTupleApproximator
from env2048 import Game2048Env
import os
# import gc


# GLOBAL_TD_MCTS = None
# GLOBAL_ROOT = None
board_size = 4

TUPLES = [
    # (a)
    [0, 1, 2, 4, 5, 6],   
    # (b)
    [1, 2, 5, 6, 9, 13],
    # (c)
    [0, 1, 2, 3, 4, 5],
    # (d)
    [0, 1, 5, 6, 7, 10],
    # (e)
    [0, 1, 2, 5, 9, 10],
    # (f)
    [0, 1, 5, 9, 13, 14],
    # (g)
    [0, 1, 5, 8, 9, 13],
    # (h)
    [0, 1, 2, 4, 6, 10]
]

patterns = [[(i // board_size, i % board_size) for i in pattern] for pattern in TUPLES]


# approximator = NTupleApproximator(board_size=board_size, patterns=patterns)

approximator = None
td_mcts = None
cur_score = 0
mem = 5



def init_model(score):
    global approximator, td_mcts, cur_score, mem

    if approximator is None:
        output = "weights_bak.npy"
        if not os.path.exists(output):
            file_id = "1Fe6jSso5E0WpwJG0BjWetKPz2R8NhG9o"
            url = f"https://drive.google.com/uc?id={file_id}"
            import gdown
            gdown.download(url, output, quiet=False)

        approximator = NTupleApproximator(board_size=4, patterns=patterns)

        weights_array = np.load("weights_bak.npy", allow_pickle=True)
        approximator.weights = [defaultdict(float, w) for w in weights_array]
        print("weights loaded and model initialized.")
    if score < mem:
        print("[Reset]")
        td_mcts = TD_MCTS(Game2048Env(), approximator, iterations=100, exploration_constant=2, rollout_depth=0, gamma=1.0)
        cur_score = 0

    return

    



def get_action(env, score):
    global approximator, td_mcts, cur_score, mem

    init_model(score)

    if score > cur_score + 5000:
        cur_score += 5000
        print("=" * 30) 
        print(f"Score {cur_score} reached")
        print(env)
        print("=" * 30) 
    elif score < cur_score:
        print("Reset")
        cur_score = 0

    state = env
    root = TD_MCTS_Node(state, score)

    if score < 20000:
        td_mcts = TD_MCTS(Game2048Env(), approximator, iterations=20, exploration_constant=1.4, rollout_depth=0, gamma=1.0)
        for i in range(20):
            td_mcts.run_simulation(root)
            best_action, distribution = td_mcts.best_action_distribution(root)
    elif score < 40000:
        td_mcts = TD_MCTS(Game2048Env(), approximator, iterations=100, exploration_constant=2, rollout_depth=0, gamma=1.0)
        for i in range(100):
            td_mcts.run_simulation(root)
            best_action, distribution = td_mcts.best_action_distribution(root)
    else:
        td_mcts = TD_MCTS(Game2048Env(), approximator, iterations=500, exploration_constant=2, rollout_depth=0, gamma=1.0)
        for i in range(500):
            td_mcts.run_simulation(root)
            best_action, distribution = td_mcts.best_action_distribution(root)
            # if i > 100 and distribution[best_action] > 0.8:
            #     break
    
    mem = score
    best_action, _ = td_mcts.best_action_distribution(root)
    return best_action
