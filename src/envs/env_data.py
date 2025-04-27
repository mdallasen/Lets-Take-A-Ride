import osmnx as ox 
import os 

# nodes, edges = ox.graph_to_gdfs(map)

# node_features = nodes[['y', 'x', 'highway', 'street_count', 'ref']]
# edge_features = edges[['osmid', 'highway', 'maxspeed', 'oneway', 'reversed', 'length',
    # 'lanes', 'service', 'access', 'bridge', 'tunnel',
    # 'width', 'junction']]

def oms_data(show_graph=False):
    """
    Set up the environment for OSMnx to work with OpenStreetMap data.
    This function configures the data folder and initializes OSMnx settings.
    """

    data_folder = os.path.join(os.getcwd(), "data")
    cache_folder = os.path.join(os.getcwd(), "data")

    ox.settings.log_console = True
    ox.settings.use_cache = True
    ox.settings.data_folder = data_folder
    ox.settings.cache_folder = cache_folder

    north, south, east, west = 40.7900, 40.7700, -73.9400, -73.9600

    map = ox.graph_from_bbox(north, south, east, west, network_type='drive_service')

    if show_graph:
        ox.plot_graph(map)

    return map