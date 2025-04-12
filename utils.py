import copy
import random
import math
import numpy as np
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt
import pickle

from env2048 import Game2048Env

# ---------------- Transformations (Symmetries) ----------------
def rot90(coords, size=4):
    return [(y, size - 1 - x) for (x, y) in coords]

def rot180(coords, size=4):
    return rot90(rot90(coords, size), size)

def rot270(coords, size=4):
    return rot90(rot180(coords, size), size)

def reflect_horizontal(coords, size=4):
    return [(x, size - 1 - y) for (x, y) in coords]

def reflect_vertical(coords, size=4):
    return [(size - 1 - x, y) for (x, y) in coords]



class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.symmetry_patterns = []
        self.weights = []

        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                if sym not in self.symmetry_patterns:
                    self.symmetry_patterns.append(sym)
                    self.weights.append(defaultdict(float))

    def generate_symmetries(self, pattern):
        syms = []
        funcs = [
            lambda x: x,
            rot90,
            rot180,
            rot270,
            reflect_horizontal,
            lambda x: rot90(reflect_horizontal(x)),
            lambda x: rot180(reflect_horizontal(x)),
            lambda x: rot270(reflect_horizontal(x))
        ]
        for f in funcs:
            transformed = tuple(sorted(f(pattern)))
            if transformed not in syms:
                syms.append(transformed)
        return syms

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[x, y]) for (x, y) in coords)

    def value(self, board):
        total = 0.0
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            total += self.weights[i][feature]
        # print(total)
        return total

    def update(self, board, delta, alpha):
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            self.weights[i][feature] += alpha * (delta  / 8)

class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None, is_random_node=False, exploration_constant=1.41):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        is_random_node: whether this node is BEFORE random tile placement
        exploration_constant: the exploration parameter (c) for UCT
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.is_random_node = is_random_node
        self.c = exploration_constant  # Store exploration constant in each node
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = []

        if not self.is_random_node:
            self.untried_actions = [a for a in range(4) if self.is_move_legal(a)]
        else:
            self.untried_actions = self.get_possible_tile_placements()

    def is_move_legal(self, action):
        temp_env = Game2048Env()
        temp_env.board = self.state.copy()
        temp_env.score = self.score
        return temp_env.is_move_legal(action)

    def get_possible_tile_placements(self):
        empty_cells = list(zip(*np.where(self.state == 0)))
        possible_placements = []
        for cell in empty_cells:
            possible_placements.append((cell, 2))  # 90% chance
            possible_placements.append((cell, 4))  # 10% chance
        return possible_placements

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_uct_value(self, parent_visits, v_norm=1.0):
        """
        UCT formula:
            UCT = (average_value / v_norm) + c * sqrt( log(parent_visits) / visits )

        v_norm can be dynamically computed by the parent to normalize exploitation.
        """
        if self.visits == 0:
            return float('inf')

        average_value = self.total_value / self.visits
        normalized_value = average_value / 25000
        exploration_term = self.c * math.sqrt(math.log(parent_visits) / self.visits)

        # print("Value1: ", normalized_value)
        # print("Value2: ", exploration_term)

        return normalized_value + exploration_term


class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=0, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        if node.is_random_node:
            placements = list(node.children.keys())
            weights = [0.9 if placement[1] == 2 else 0.1 for placement in placements]
            selected_placement = random.choices(placements, weights=weights)[0]
            return node.children[selected_placement]
        else:
        
            v_norm_candidates = []

            for child in node.children.values():
                if child.visits > 0:
                    avg_value = child.total_value / child.visits
                    v_norm_candidates.append(avg_value)

            if len(v_norm_candidates) == 0:
                # Fallback if no child visited
                v_norm = 1200
            else:
                v_norm = max(v_norm_candidates)

            best_child = None
            best_value = -float('inf')
            for child in node.children.values():
                uct_value = child.get_uct_value(node.visits, v_norm=v_norm)
                if uct_value > best_value:
                    best_value = uct_value
                    best_child = child

            return best_child

    def expand(self, node):
        if node.is_random_node:
            untried = node.untried_actions[:]

            # tile_placement = random.choice(untried)
            untried = node.untried_actions[:]
            weights = [0.9 if v == 2 else 0.1 for (_, v) in untried]
            tile_placement = random.choices(untried, weights=weights)[0]

            (x, y), value = tile_placement
            # print
            new_state = node.state.copy()
            new_state[x, y] = value

            is_duplicate = any(np.array_equal(child.state, new_state) for child in node.children.values())
            if is_duplicate:
                print("[WARNING] duplicate child detected!!")
                node.untried_actions.remove(tile_placement)
                # continue

            new_score = node.score
            child_node = TD_MCTS_Node(
                new_state, new_score,
                parent=node,
                action=None,
                is_random_node=False,
                exploration_constant=self.c
            )
            node.children[tile_placement] = child_node
            node.untried_actions.remove(tile_placement)

            return child_node

            # return None

        else:
            # Expand an action
            action = random.choice(node.untried_actions)
            sim_env = self.create_env_from_state(node.state, node.score)

            # Execute action without adding random tile
            if action == 0:
                sim_env.move_up()
            elif action == 1:
                sim_env.move_down()
            elif action == 2:
                sim_env.move_left()
            elif action == 3:
                sim_env.move_right()

            new_state = sim_env.board.copy()
            new_score = sim_env.score
            reward = new_score - node.score

            child_node = TD_MCTS_Node(
                new_state, new_score,
                parent=node,
                action=action,
                is_random_node=True,
                exploration_constant=self.c
            )
            child_node.reward = reward
            node.children[action] = child_node
            node.untried_actions.remove(action)

        return child_node

    def rollout(self, node, depth):
        total_reward = 0
        discount = 1.0
        sim_env = self.create_env_from_state(node.state, node.score)

        for _ in range(depth):
            if sim_env.is_game_over():
                break

            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)

            old_score = sim_env.score
            _, reward, done, _ = sim_env.step(action)
            actual_reward = reward - old_score
            total_reward += discount * actual_reward
            discount *= self.gamma

        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_moves:
            return 0

        values = []
        pre_random_states = []
        for a in legal_moves:
            tmp_env = copy.deepcopy(sim_env)
            score_before = tmp_env.score
            if a == 0:
                tmp_env.move_up()
            elif a == 1:
                tmp_env.move_down()
            elif a == 2:
                tmp_env.move_left()
            elif a == 3:
                tmp_env.move_right()
            reward_sim = tmp_env.score - score_before
            s_candidate = tmp_env.board.copy()
            # print(approximator.value(s_candidate))
            value_est = reward_sim + self.gamma * self.approximator.value(s_candidate)
            values.append(value_est)
            pre_random_states.append(s_candidate)

        final_value = max(values)
        total_reward += discount * final_value

        return total_reward


    def backpropagate(self, node, reward):
        discount = 1.0
        while node is not None:
            node.visits += 1
            node.total_value += discount * reward
            if hasattr(node, 'reward'):
                reward += node.reward

            discount *= self.gamma
            node = node.parent



    def run_simulation(self, root):
        node = root

        # Selection
        d = 0
        while node.fully_expanded():
            if not node.children:
                break
            d += 1
            node = self.select_child(node)

        # print("Depth: ", d)

        # Expansion
        if not node.fully_expanded():
            node = self.expand(node)

        # Rollout
        if node.is_random_node:
            # For random nodes, use the approximator directly
            rollout_value = self.approximator.value(node.state)
        else:
            # For action nodes, perform rollout
            rollout_value = self.rollout(node, self.rollout_depth)

        # Backpropagation
        self.backpropagate(node, rollout_value)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None

        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0

        return best_action, distribution

    def search(self, initial_state, initial_score):
        root = TD_MCTS_Node(
            initial_state, initial_score,
            is_random_node=False,
            exploration_constant=self.c
        )

        for _ in range(self.iterations):
            self.run_simulation(root)

        return self.best_action_distribution(root)
    

