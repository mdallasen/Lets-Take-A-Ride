import math

def edistance(start, end, map): 

    start_lat, start_long = map.nodes[start['y']], map.nodes[start['x']]
    end_lat, end_long = map.nodes[end['y']], map.nodes[end['x']]

    return math.sqrt((end_lat - start_lat) ** 2 + (end_long - start_long) ** 2)

