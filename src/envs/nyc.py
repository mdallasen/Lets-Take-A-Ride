import osmnx as ox 
import os 

data_folder = os.path.join(os.getcwd(), "src", "data")
ox.config(log_console=True, use_cache=True, data_folder=data_folder)
map = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')

ox.plot_graph(map)