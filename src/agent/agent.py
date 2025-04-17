import gym 
from gym import spaces
import numpy as np 
import random 
import networkx as nx 
import osmnx as ox 
from utils import edistance

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

    def reset(self):
        """
        Resets the agent's state by selecting a random starting node and returning its index.
        """
        self.current_node = random.choice(list(self.map.nodes()))
        node_index = self.node_to_index[self.current_node]
        return node_index
    
    def step(self, action):
        """
        Updates the agent's state by moving to the next node, checks if the goal is reached, 
        assigns rewards or penalties based on movement, and returns the updated state and reward.
        """     
        # Moving to nearest node
        neighbors = list(self.map.neighbors(self.current_node))
        neighbors = sorted(neighbors, key=lambda n: (self.map.nodes[n]['y'], self.map.nodes[n]['x']))
        next_node = neighbors[action % len(neighbors)]
        self.previous_node = self.current_node
        self.current_node = next_node
        self.visited_nodes.append(next_node)
        
        # Checking if reached end goal yet
        self.done = self.current_node == self.goal_node
        if self.done:
            self.reward += 1000
        else:
            self.reward -= 1000

        # Calculating distance from end goal
        current_distance = edistance(self.current_node, self.goal_node, self.map)
        next_distance = edistance(next_node, self.goal_node, self.map)

        # Checking if moving towards goal        
        if next_distance > current_distance:
            self.reward -= 5 
        self.steps_taken += 1

        return self.node_to_index[self.current_node], self.reward, self.done, {}
    
    def render(self, graph = True): 
        """
        Prints the current node and optionally displays the path taken by the agent on the map.
        """
        print(self.current_node)
        if graph: 
            ox.plot_graph_route(self.map, self.visited_nodes)

    
        
