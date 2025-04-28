import numpy as np

def edistance(start, end, graph):

    lat1, lon1 = graph.nodes[start]["y"], graph.nodes[start]["x"]
    lat2, lon2 = graph.nodes[end]["y"], graph.nodes[end]["x"]
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

def encode_state(current_node_index, goal_node_index, state_space):
    # Ensure the indices are integers
    current_node_index = int(current_node_index)
    goal_node_index = int(goal_node_index)

    encoded_state = np.zeros(state_space)
    encoded_state[current_node_index] = 1 
    encoded_state[goal_node_index] = 2
    return encoded_state