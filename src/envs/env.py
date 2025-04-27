import gymnasium as gym
from gymnasium import spaces
import random
import networkx as nx
from utils.helper import edistance, encode_state
class GraphEnv(gym.Env):
    """Goal-conditioned graph navigation"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, graph: nx.Graph, max_steps: int = 500):
        super().__init__()

        self.map = graph
        self.nodes = list(graph.nodes())
        self.node_to_index = {n: i for i, n in enumerate(self.nodes)}

        self.action_space = spaces.Discrete(3)             
        self.observation_space = spaces.Discrete(len(self.nodes))

        self.max_steps = max_steps
        self.current_node = None
        self.goal_nodes = []
        self.steps_taken = 0
        self.done = False
        self.reward = 0.0
        self.state_space = len(self.nodes)
        self.visited_goals = set() 

    def reset(self, *, seed=None, options=None, specific_goal=None):
        super().reset(seed=seed)

        if specific_goal:  # If a specific goal is passed in (for testing)
            self.goal_node = specific_goal
        else:
            self.goal_nodes = random.sample(self.nodes, 3)
            self.goal_node = random.choice(self.goal_nodes)

        self.current_node = random.choice(self.nodes)

        self.visited_goals = set()
        self.steps_taken = 0
        self.done = False
        self.reward = 0.0

        return self.node_to_index[self.current_node], {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished — call reset() before step().")

        neighbours = list(self.map.neighbors(self.current_node))

        if not neighbours:
            self.done = True
            return self.node_to_index[self.current_node], -10.0, True, False, {}

        neighbours.sort(key=lambda n: (self.map.nodes[n]["y"], self.map.nodes[n]["x"]))
        next_node = neighbours[action % len(neighbours)]     # wraps safely for degree < 3

        unvisited_goals = [g for g in self.goal_nodes if g not in self.visited_goals]
        if not unvisited_goals:
            self.done = True
            return self.node_to_index[self.current_node], 0.0, True, False, {}

        cur_d = min(edistance(self.current_node, g, self.map) for g in unvisited_goals)
        nxt_d = min(edistance(next_node, g, self.map) for g in unvisited_goals)

        reward = -1.0  # step cost
        reward += 1.0 if nxt_d < cur_d else -1.0   # moving closer / farther

        self.current_node = next_node
        self.steps_taken += 1

        if self.current_node in unvisited_goals:
            self.visited_goals.add(self.current_node)
            reward += 500.0

            if len(self.visited_goals) == len(self.goal_nodes): 
                reward += 1000
                self.done = True

        truncated = self.steps_taken >= self.max_steps
        if truncated:
            self.done = True

        state = encode_state(self.node_to_index[self.current_node], self.node_to_index[self.goal_node], self.state_space)

        return state, reward, self.done, truncated, {}

    def render(self, mode="human"):
        print(f"{self.current_node} → goal {self.goal_nodes}  step {self.steps_taken}")

    def close(self):
        pass
