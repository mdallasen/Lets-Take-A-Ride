from envs.env import GraphEnv
from envs.env_data import oms_data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import tensorflow as tf
import os
import shutil
from PIL import Image

def visualize_trip(model, env, visited, node_size=10):
    """
    Visualizes an existing trip from start to goal using the visited path.
    """
    pos = {node: (env.map.nodes[node]['x'], env.map.nodes[node]['y']) for node in env.map.nodes}

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(env.map, pos, node_size=node_size, edge_color='gray', alpha=0.3, ax=ax)

    # Zoom into path region
    visited_x = [pos[n][0] for n in visited]
    visited_y = [pos[n][1] for n in visited]
    padding = 0.002
    ax.set_xlim(min(visited_x) - padding, max(visited_x) + padding)
    ax.set_ylim(min(visited_y) - padding, max(visited_y) + padding)

    # Add goal flags
    for goal in env.goal_nodes:
        goal_x, goal_y = pos[goal]
        ax.text(goal_x, goal_y, "GOAL", fontsize=16, ha='center', va='center', zorder=5)

    text_handle = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                          verticalalignment='top', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))

    emoji_handle = None
    cumulative_distance = 0.0

    for i in range(1, len(visited)):
        edge = (visited[i - 1], visited[i])
        nx.draw_networkx_edges(env.map, pos, edgelist=[edge], edge_color='blue', width=2, ax=ax)

        if emoji_handle:
            emoji_handle.remove()

        x, y = pos[visited[i]]
        emoji_handle = ax.text(x, y, "CAR", fontsize=16, ha='center', va='center', zorder=6)

        x1, y1 = pos[visited[i-1]]
        x2, y2 = pos[visited[i]]
        cumulative_distance += ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        text_handle.set_text(f"Distance traveled: {cumulative_distance:.4f}")
        plt.pause(0.3)

    plt.title("Driving from Start to Goal")
    plt.tight_layout()
    plt.show()

def visual_gif(model, env, visited, node_size=10, gif_path="trip.gif", max_frames=50):
    """
    Visualizes an existing trip and saves it to a GIF with at most max_frames frames.
    """
    pos = {node: (env.map.nodes[node]['x'], env.map.nodes[node]['y']) for node in env.map.nodes}

    frame_dir = "_trip_frames"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)

    indices = np.linspace(1, len(visited) - 1, min(max_frames, len(visited) - 1)).astype(int)

    frame_paths = []

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(env.map, pos, node_size=node_size, edge_color='gray', alpha=0.3, ax=ax)

    visited_x = [pos[n][0] for n in visited]
    visited_y = [pos[n][1] for n in visited]
    padding = 0.002
    ax.set_xlim(min(visited_x) - padding, max(visited_x) + padding)
    ax.set_ylim(min(visited_y) - padding, max(visited_y) + padding)

    for goal in env.goal_nodes:
        goal_x, goal_y = pos[goal]
        ax.text(goal_x, goal_y, "GOAL", fontsize=16, ha='center', va='center', zorder=5)

    path_line, = ax.plot([], [], color='blue', linewidth=2)
    emoji_handle = ax.text(*pos[visited[0]], "CAR", fontsize=16, ha='center', va='center', zorder=6)
    text_handle = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                          verticalalignment='top', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))

    cumulative_distances = [0.0]
    for i in range(1, len(visited)):
        x1, y1 = pos[visited[i-1]]
        x2, y2 = pos[visited[i]]
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        cumulative_distances.append(cumulative_distances[-1] + distance)

    for idx, i in enumerate(indices):
        path_x, path_y = zip(*[pos[n] for n in visited[:i+1]])
        path_line.set_data(path_x, path_y)

        x, y = pos[visited[i]]
        emoji_handle.set_position((x, y))

        text_handle.set_text(f"Distance traveled: {cumulative_distances[i]:.4f}")

        frame_file = os.path.join(frame_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_file, dpi=80)
        frame_paths.append(frame_file)

    plt.close(fig)

    frames = [Image.open(fp) for fp in frame_paths]
    frames[0].save(gif_path, format='GIF', save_all=True, append_images=frames[1:], duration=300, loop=0)
    print(f"GIF saved to {gif_path}")
    shutil.rmtree(frame_dir)

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
