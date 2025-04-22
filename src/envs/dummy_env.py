import tensorflow as tf
import numpy as np
import random
import gym
import networkx as nx

class DummyGraphEnv(gym.Env):
    def __init__(self, map):
        self.map = map
        self.nodes = list(self.map.nodes())
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        self.action_space = gym.spaces.Discrete(3)
        self.state_space = len(self.nodes)
        self.goal_node = 2

    def reset(self):
        self.current_node = random.choice(self.nodes)
        return self.node_to_index[self.current_node], {}

    def step(self, action):
        neighbors = list(self.map.neighbors(self.current_node))
        next_node = self.current_node
        if neighbors:
            next_node = random.choice(neighbors)
        reward = 1 if next_node == self.goal_node else -0.1
        terminated = next_node == self.goal_node
        truncated = False
        self.current_node = next_node
        return self.node_to_index[next_node], reward, terminated, truncated, {}
