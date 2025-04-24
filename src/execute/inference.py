from envs.env import GraphEnv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox 

def visualize_episode(model, env=None):
    """
    Runs one episode using the trained model and visualizes the path on the graph.
    """
    if env is None:
        env = GraphEnv()

    state = env.reset()[0]
    done = False
    visited = [env.current_node]

    while not done:
        # Get action from model
        state_tensor = np.expand_dims(state, axis=0).astype(np.float32)
        probs, _ = model(state_tensor)
        action = np.argmax(probs[0].numpy())

        next_state, reward, terminated, truncated, _ = env.step(action)
        visited.append(env.current_node)
        state = next_state
        done = terminated or truncated

    # Plot the graph
    fig, ax = ox.plot_graph(env.map, show=False, close=False)

    # Highlight the visited path
    path_edges = list(zip(visited[:-1], visited[1:]))
    nx.draw_networkx_nodes(env.map, pos=nx.get_node_attributes(env.map, 'x'), nodelist=visited, node_color='red', node_size=10, ax=ax)
    nx.draw_networkx_edges(env.map, pos=nx.get_node_attributes(env.map, 'x'), edgelist=path_edges, edge_color='blue', width=2, ax=ax)

    # Show the plot
    plt.title("Agent's Path During One Episode")
    plt.show()

def visualize_data(total_rewards):
    """
    Visualizes the total reward per episode across training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, label='Total Reward per Episode')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Progress Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

