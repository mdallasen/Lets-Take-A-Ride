from envs.env import GraphEnv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox 
import tensorflow as tf
import emoji
import os
import shutil
from PIL import Image



def visualize_trip(model, env=None, node_size=10):
    """
    Visualizes a full trip from start to goal using the trained model.
    Animates a car emoji moving across the map and shows live distance traveled.
    """
    if env is None:
        env = GraphEnv()

    state = env.reset()[0]


    done = False
    visited = [env.current_node]
    total_distance = 0.0
    goal_node = env.goal_node
    check = False

    while not done:
        if env.current_node == env.goal_node:
            if check == False:
                print("🚨 Start and goal are the same! No trip needed.")
            break

        # Model chooses action
        state_tensor = tf.one_hot([state], depth=env.state_space, dtype=tf.float32) 
        q_values = model(state_tensor)
        action = np.argmax(q_values[0].numpy())
        check = True


        if env.done:
            break

        next_state, reward, terminated, truncated, _ = env.step(action)
        visited.append(env.current_node)

        # Increment distance
        x1, y1 = env.map.nodes[visited[-2]]['x'], env.map.nodes[visited[-2]]['y']
        x2, y2 = env.map.nodes[visited[-1]]['x'], env.map.nodes[visited[-1]]['y']
        total_distance += ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        state = next_state
        

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

    # Add static flag emoji at goal
    goal_x, goal_y = pos[goal_node]
    ax.text(goal_x, goal_y, "Goal", fontsize=16, ha='center', va='center', zorder=5)

    # Live distance text
    text_handle = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                          verticalalignment='top', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))

    emoji_handle = None

    for i in range(1, len(visited)):
        edge = (visited[i - 1], visited[i])
        nx.draw_networkx_edges(env.map, pos, edgelist=[edge], edge_color='blue', width=2, ax=ax)

        if emoji_handle:
            emoji_handle.remove()

        # Add new car emoji
        x, y = pos[visited[i]]
        emoji_handle = ax.text(x, y, "Car", fontsize=16, ha='center', va='center', zorder=6)

        # Update live distance
        text_handle.set_text(f"Distance traveled: {total_distance:.4f}")
        plt.pause(0.3)

    plt.title("Driving from Start to Goal")
    plt.tight_layout()
    plt.show()

def visual_gif(model, env=None, node_size=10, gif_path="trip.gif", max_frames=50):
    """
    Visualizes a trip and saves it to a GIF with at most max_frames frames.
    """
    if env is None:
        env = GraphEnv()

    state = env.reset()[0]

    if env.current_node == env.goal_node:
        print("🚨 Start and goal are the same! No trip needed.")
        return

    visited = [env.current_node]
    cumulative_distances = [0.0]  # Store cumulative distance at each step
    goal_node = env.goal_node

    done = False
    while not done:
        if env.current_node == env.goal_node:
            break

        # Model chooses action
        state_tensor = tf.one_hot([state], depth=env.state_space, dtype=tf.float32)
        q_values = model(state_tensor)
        action = np.argmax(q_values[0].numpy())

        if env.done:
            break

        next_state, reward, terminated, truncated, _ = env.step(action)
        visited.append(env.current_node)

        # Update distance
        x1, y1 = env.map.nodes[visited[-2]]['x'], env.map.nodes[visited[-2]]['y']
        x2, y2 = env.map.nodes[visited[-1]]['x'], env.map.nodes[visited[-1]]['y']
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        cumulative_distances.append(cumulative_distances[-1] + distance)

        state = next_state

    # Precompute node positions
    pos = {node: (env.map.nodes[node]['x'], env.map.nodes[node]['y']) for node in env.map.nodes}

    # Frame output folder
    frame_dir = "_trip_frames"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)

    # Choose frame indices evenly spaced
    indices = np.linspace(1, len(visited) - 1, min(max_frames, len(visited)-1)).astype(int)

    frame_paths = []

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(env.map, pos, node_size=node_size, edge_color='gray', alpha=0.3, ax=ax)

    # Static elements
    visited_x = [pos[n][0] for n in visited]
    visited_y = [pos[n][1] for n in visited]
    padding = 0.002
    ax.set_xlim(min(visited_x) - padding, max(visited_x) + padding)
    ax.set_ylim(min(visited_y) - padding, max(visited_y) + padding)

    goal_x, goal_y = pos[goal_node]
    ax.text(goal_x, goal_y, "🏁", fontsize=16, ha='center', va='center', zorder=5)

    # Dynamic elements
    path_line, = ax.plot([], [], color='blue', linewidth=2)
    emoji_handle = ax.text(*pos[visited[0]], "CAR", fontsize=16, ha='center', va='center', zorder=6)
    text_handle = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                          verticalalignment='top', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))

    for idx, i in enumerate(indices):
        # Update path
        path_x, path_y = zip(*[pos[n] for n in visited[:i+1]])
        path_line.set_data(path_x, path_y)

        # Update car position
        x, y = pos[visited[i]]
        emoji_handle.set_position((x, y))

        # Update distance text
        text_handle.set_text(f"Distance traveled: {cumulative_distances[i]:.4f}")

        # Save frame
        frame_file = os.path.join(frame_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_file, dpi=80)
        frame_paths.append(frame_file)

    plt.close(fig)

    # Save to GIF
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

