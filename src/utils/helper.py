import math

# THINGS TO DO: 
    # Add any helper funcitons you use here (DON'T DELETE OR ALTER ANY OTHERS)
    # Change to manhattan distance ???? 

# def edistance(start, end, map): 

#     start_lat, start_long = map.nodes[start['y']], map.nodes[start['x']]
#     end_lat, end_long = map.nodes[end['y']], map.nodes[end['x']]

#     return math.sqrt((end_lat - start_lat) ** 2 + (end_long - start_long) ** 2)
def edistance(start, end, graph):
    """
    Euclidean distance between two nodes using their lat/lon.
    """
    lat1, lon1 = graph.nodes[start]["y"], graph.nodes[start]["x"]
    lat2, lon2 = graph.nodes[end]["y"], graph.nodes[end]["x"]
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
