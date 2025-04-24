import osmnx as ox 
import os 

print(ox.__version__)

def oms_data(show_graph = False):
    """
    Set up the environment for OSMnx to work with OpenStreetMap data.
    This function configures the data folder and initializes OSMnx settings.
    """

    data_folder = os.path.join(os.getcwd(), "data")
    cache_folder = os.path.join(os.getcwd(), "data")
    # ox.settings(log_console=True, use_cache=True, data_folder=data_folder, cache_folder=cache_folder)

    ox.settings.log_console = True
    ox.settings.use_cache = True
    ox.settings.data_folder = data_folder
    ox.settings.cache_folder = cache_folder
    place_name = 'Manhattan, New York, USA'
    map = ox.graph_from_place(place_name, network_type='drive_service')
    # nodes, edges = ox.graph_to_gdfs(map)
    
    # node_features = nodes[['y', 'x', 'highway', 'street_count', 'ref']]
    # edge_features = edges[['osmid', 'highway', 'maxspeed', 'oneway', 'reversed', 'length',
       # 'lanes', 'service', 'access', 'bridge', 'tunnel',
       # 'width', 'junction']]
    def load_district_graph(district_name: str, network_type: str = "drive_service"):
        """
        district_name examples
        • "Harlem, Manhattan, New York, USA"
        • "Upper East Side, Manhattan, New York, USA"
        • "Brooklyn Heights, Brooklyn, New York, USA"
        """
        ox.settings.log_console = True
        polygon = ox.geocode_to_gdf(district_name).geometry.iloc[0]  # shapely Polygon
        G = ox.graph_from_polygon(polygon, network_type=network_type)
        return G


    #G_harlem = load_district_graph("Harlem, Manhattan, New York, USA")
    nodes, edges = ox.graph_to_gdfs(map)
    #nodes, edges = ox.graph_to_gdfs(G_harlem)
    sample_nodes = nodes.head(20)
    subgraph = map.subgraph(sample_nodes.index)

    if show_graph: 
        ox.plot_graph(subgraph)

    return subgraph