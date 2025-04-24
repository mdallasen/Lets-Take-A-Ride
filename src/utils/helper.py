import math

def edistance(start, end, graph):

    lat1, lon1 = graph.nodes[start]["y"], graph.nodes[start]["x"]
    lat2, lon2 = graph.nodes[end]["y"], graph.nodes[end]["x"]
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
