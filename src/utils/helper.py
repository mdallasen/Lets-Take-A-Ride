import math

def edistance(start, end, graph):

    lat1, lon1 = graph.nodes[start]["y"], graph.nodes[start]["x"]
    lat2, lon2 = graph.nodes[end]["y"], graph.nodes[end]["x"]
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

def encode_state(current_node, goal_node, state_space):

    state = [0] * (state_space * 2)  
    state[current_node] = 1
    state[state_space + goal_node] = 1
    
    return state