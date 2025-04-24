# import gymnasium as gym 
# from gymnasium import spaces
# import numpy as np 
# import random 
# import networkx as nx 
# import osmnx as ox 
# from utils.helper import edistance

# # THINGS TO DO: 
#     # Better define the action space r.e. nodes 
#     # Need to confirm that our custom environment is set up properly!!!!!! 
#     # Edges are treated equally, need to assign a higher weight the longer they are
#     # We don't account for edge characteristics e.g. highway vs road vs inaccesible road, need to create a list of valid neighbours
#     # Need to improve the reward structure within steps to account for more complex scenarios and distance away

# class GraphEnv(gym.Env):
#     def __init__(self, map):
#         super(GraphEnv, self).__init__() 
#         self.map = map
#         self.goal_node = random.choice(list(map.nodes()))
#         self.nodes = list(map.nodes())
#         self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
#         self.done = False
#         self.visited_nodes = []
#         self.action_space = spaces.Discrete(3)  
#         self.reward = 0
#         self.steps_taken = 0
#         self.state_space = len(self.nodes)

#     def reset(self):
#         """
#         Resets the agent's state by selecting a random starting node and returning its index.
#         """
#         self.current_node = random.choice(list(self.map.nodes()))
#         node_index = self.node_to_index[self.current_node]
#         return node_index, {}
    
#     def step(self, action):
#         """
#         Updates the agent's state by moving to the next node, checks if the goal is reached, 
#         assigns rewards or penalties based on movement, and returns the updated state and reward.
#         """     
#         # Determing the neighbours of the current nodes
#         neighbors = list(self.map.neighbors(self.current_node))
#         neighbors = sorted(neighbors, key=lambda n: (self.map.nodes[n]['y'], self.map.nodes[n]['x']))

#         # Mapping the action to a specific neighbour
#         if action == 0:  
#             next_node = neighbors[0]
#         elif action == 1:  
#             next_node = neighbors[1] if len(neighbors) > 1 else neighbors[0]
#         elif action == 2:  
#             next_node = neighbors[2] if len(neighbors) > 2 else neighbors[0]
        
#         # Making a move based on the defined actions 
#         current_distance = edistance(self.current_node, self.goal_node, self.map)
#         next_node = neighbors[action % len(neighbors)]
#         next_distance = edistance(next_node, self.goal_node, self.map)
#         self.previous_node = self.current_node
#         self.current_node = next_node
#         self.visited_nodes.append(next_node)
        
#         # Checking if reached end goal yet
#         self.done = self.current_node == self.goal_node
#         if self.done:
#             self.reward = 1000
#         else:
#             self.reward = -1

#         # Checking if moving towards goal        
#         if next_distance > current_distance:
#             self.reward -= 5 
#         self.steps_taken += 1

#         terminated = self.done
#         truncated = False 
#         return self.node_to_index[self.current_node], self.reward, terminated, truncated, {}
import gymnasium as gym
from gymnasium import spaces
import random
import networkx as nx
from utils.helper import edistance


class GraphEnv(gym.Env):
    """Goal-conditioned graph navigation"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, graph: nx.Graph, max_steps: int = 500):
        super().__init__()

        # --- graph & lookup tables
        self.map = graph
        self.nodes = list(graph.nodes())
        self.node_to_index = {n: i for i, n in enumerate(self.nodes)}

        # --- gym spaces
        self.action_space = spaces.Discrete(3)              # we’ll map 0-2 to available neighbours
        self.observation_space = spaces.Discrete(len(self.nodes))

        # --- episode-level state
        self.max_steps = max_steps
        self.current_node = None
        self.goal_node = None
        self.steps_taken = 0
        self.done = False
        self.reward = 0.0
        self.state_space = len(self.nodes)
        print('djfkdjdfkdjfkd',self.state_space)

    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.goal_node = random.choice(self.nodes)
        self.current_node = random.choice(self.nodes)
        self.steps_taken = 0
        self.done = False
        self.reward = 0.0

        return self.node_to_index[self.current_node], {}

    # ------------------------------------------------------------
    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished — call reset() before step().")

        neighbours = list(self.map.neighbors(self.current_node))

        # Handle dead-ends
        if not neighbours:
            self.done = True
            return self.node_to_index[self.current_node], -10.0, True, False, {}

        # Stable ordering so action→neighbour mapping is deterministic
        neighbours.sort(key=lambda n: (self.map.nodes[n]["y"], self.map.nodes[n]["x"]))
        next_node = neighbours[action % len(neighbours)]     # wraps safely for degree < 3

        # --- reward shaping -------------------------------------------------
        cur_d = edistance(self.current_node, self.goal_node, self.map)
        nxt_d = edistance(next_node, self.goal_node, self.map)

        reward = -1.0                       # step cost
        reward += 1.0 if nxt_d < cur_d else -1.0   # moving closer / farther

        self.current_node = next_node
        self.steps_taken += 1

        if self.current_node == self.goal_node:
            reward += 1000.0
            self.done = True

        truncated = self.steps_taken >= self.max_steps
        if truncated:
            self.done = True

        return self.node_to_index[self.current_node], reward, self.done, truncated, {}

    # ------------------------------------------------------------
    def render(self, mode="human"):
        print(f"{self.current_node} → goal {self.goal_node}  step {self.steps_taken}")

    def close(self):
        pass
