import osmnx as ox 
import os 

def setup_environment(show_graph = True):
    """
    Set up the environment for OSMnx to work with OpenStreetMap data.
    This function configures the data folder and initializes OSMnx settings.
    """

    data_folder = os.path.join(os.getcwd(), "src", "data")
    cache_folder = os.path.join(os.getcwd(), "src", "data")
    ox.config(log_console=True, use_cache=True, data_folder=data_folder, cache_folder=cache_folder)
    map = ox.graph_from_place('Manhattan, New York, USA', network_type='drive_service')
    nodes, edges = ox.graph_to_gdfs(map)
    
    node_features = nodes[['osmid', 'x', 'y', 'street_count', 'degree']]
    edge_features = edges[['osmid', 'highway', 'length', 'oneway', 'lanes', 'maxspeed', 'traffic_signals', 'surface', 'access']]
    
    if show_graph: 
        ox.plot_graph(map)

    return node_features, edge_features
    

setup_environment()