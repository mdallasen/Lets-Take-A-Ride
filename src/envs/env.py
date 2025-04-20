import gym 
from gym import spaces
import numpy as np 
import random 
import networkx as nx 
import osmnx as ox 
from utils.helper import edistance

# THINGS TO DO: 
    # Edges are treated equally, need to assign a higher weight the longer they are
    # We don't account for edge characteristics e.g. highway vs road vs inaccesible road, need to create a list of valid neighbours
    # Need to improve the reward structure within steps to account for more complex scenarios and distance away

class GraphEnv(gym.Env):
    def __init__(self, map):
        super(GraphEnv, self).__init__() 
        self.map = map
        self.goal_node = random.choice(list(map.nodes()))
        self.nodes = list(map.nodes())
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        self.done = False
        self.visited_nodes = []
        self.action_space = spaces.Discrete(3)  
        self.reward = 0
        self.steps_taken = 0
        self.state_space = len(self.nodes)

    def reset(self):
        """
        Resets the agent's state by selecting a random starting node and returning its index.
        """
        self.current_node = random.choice(list(self.map.nodes()))
        node_index = self.node_to_index[self.current_node]
        return node_index, {}
    
    def step(self, action):
        """
        Updates the agent's state by moving to the next node, checks if the goal is reached, 
        assigns rewards or penalties based on movement, and returns the updated state and reward.
        """     
        # Determing the neighbours of the current nodes
        neighbors = list(self.map.neighbors(self.current_node))
        neighbors = sorted(neighbors, key=lambda n: (self.map.nodes[n]['y'], self.map.nodes[n]['x']))

        # Mapping the action to a specific neighbour
        if action == 0:  
            next_node = neighbors[0]
        elif action == 1:  
            next_node = neighbors[1] if len(neighbors) > 1 else neighbors[0]
        elif action == 2:  
            next_node = neighbors[2] if len(neighbors) > 2 else neighbors[0]
        
        # Making a move based on the defined actions 
        current_distance = edistance(self.current_node, self.goal_node, self.map)
        next_node = neighbors[action % len(neighbors)]
        next_distance = edistance(next_node, self.goal_node, self.map)
        self.previous_node = self.current_node
        self.current_node = next_node
        self.visited_nodes.append(next_node)
        
        # Checking if reached end goal yet
        self.done = self.current_node == self.goal_node
        if self.done:
            self.reward = 1000
        else:
            self.reward = -1

        # Checking if moving towards goal        
        if next_distance > current_distance:
            self.reward -= 5 
        self.steps_taken += 1

        terminated = self.done
        truncated = False 
        return self.node_to_index[self.current_node], self.reward, terminated, truncated, {}

    
        
