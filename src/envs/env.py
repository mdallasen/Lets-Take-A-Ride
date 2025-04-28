import gymnasium as gym
from gymnasium import spaces
import random
import networkx as nx
from utils.helper import edistance
class GraphEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, graph: nx.Graph, max_steps: int = 500):
        super().__init__()
        self.map = graph
        self.nodes = list(graph.nodes())
        self.node_to_index = {n: i for i, n in enumerate(self.nodes)}
        self.action_space = spaces.Discrete(4)             
        self.max_steps = max_steps
        self.steps_taken = 0
        self.done = False
        self.state_space = len(self.nodes)
        self.goal_space = 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.goal_nodes = random.sample(self.nodes, self.goal_space)
        self.goal_node = random.choice(self.goal_nodes)
        
        self.current_node = random.choice(self.nodes)
        while self.current_node == self.goal_node or not list(self.map.neighbors(self.current_node)):
            self.current_node = random.choice(self.nodes)

        self.steps_taken = 0
        self.done = False
        self.reward = 0.0

        state = (self.node_to_index[self.current_node], self.node_to_index[self.goal_node])

        return state, {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished — call reset() before step().")

        neighbours = list(self.map.neighbors(self.current_node))

        if not neighbours:
            self.done = True
            return self.node_to_index[self.current_node], -10.0, True, False, {}
        
        action_reward = []
        
        for i, next_node in enumerate(neighbours):
            edge = self.map.get_edge_data(self.current_node, next_node)
            oneway = edge.get('oneway', False)
            reversed_edge = edge.get('reversed', False)
            is_highway = self.map.nodes[next_node].get('highway', False)
            street_count = self.map.nodes[next_node].get('street_count', 0)

            # Penalize if the edge is one-way and the agent is going in the wrong direction
            if oneway and reversed_edge:
                action_reward.append(-10.0)  # Penalize for wrong direction

            # Reward for moving to a highway node and nodes with higher street count
            reward = 0
            if is_highway:
                reward += 5.0  # Encourage moving towards highways
            reward += street_count * 0.1  # Reward for higher street count

            # Calculate the distance between current node and goal node
            cur_d = edistance(self.current_node, self.goal_node, self.map)
            nxt_d = edistance(next_node, self.goal_node, self.map)

            # Penalize for moving away from the goal
            reward += 1.0 if nxt_d < cur_d else -1.0

            action_reward.append(reward)
        
        next_node = neighbours[action % len(neighbours)]

        self.current_node = next_node
        self.steps_taken += 1

        if self.current_node == self.goal_node:
            reward += 1000.0
            self.done = True

        truncated = self.steps_taken >= self.max_steps
        if truncated:
            self.done = True

        state = (self.node_to_index[self.current_node], self.node_to_index[self.goal_node])

        print(state)

        return state, reward, self.done, truncated, {}
    
    def render(self, mode="human"):
        print(f"{self.current_node} → goal {self.goal_node}  step {self.steps_taken}")

    def close(self):
        pass
