import tensorflow as tf
import numpy as np
import random
import gym
import networkx as nx

class DummyGraphEnv(gym.Env):
    def __init__(self, map):
        """
        Initializes the environment for navigation on the provided map graph.
        """
        self.map = map
        self.nodes = list(self.map.nodes())
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}

        # Action space with 3 possible actions
        self.action_space = gym.spaces.Discrete(3)
        
        # State space corresponds to the number of nodes in the map
        self.state_space = len(self.nodes)
        
        # Setting a fixed random seed for reproducibility
        random.seed(42)
        
        # Randomly select a goal node (can also be hardcoded to a specific node)
        self.goal_node = random.choice(self.nodes)

    def reset(self):
        """
        Resets the environment by choosing a random starting node.
        """
        self.current_node = random.choice(self.nodes)
        return self.node_to_index[self.current_node], {}

    def step(self, action):
        """
        Executes a step based on the action taken.
        - Finds the neighboring nodes of the current node.
        - Randomly chooses a neighboring node to transition to.
        - Returns the next state, reward, and termination condition.
        """
        neighbors = list(self.map.neighbors(self.current_node))
        next_node = self.current_node

        # If there are neighbors, move to a random one
        if neighbors:
            next_node = random.choice(neighbors)
        
        # Reward: positive for reaching the goal node, negative for any other move
        reward = 1 if next_node == self.goal_node else -0.1
        
        # Check if the goal is reached
        terminated = next_node == self.goal_node
        truncated = False  # Can be extended for time-based termination, if needed
        
        # Update the current node for the next step
        self.current_node = next_node
        
        # Return the new state (node index), reward, termination status, truncated flag, and additional info
        return self.node_to_index[next_node], reward, terminated, truncated, {}
