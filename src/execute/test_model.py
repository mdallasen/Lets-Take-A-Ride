import numpy as np
import tensorflow as tf
from utils.helper import edistance
from execute.inference import visualize_data, visualize_trip, visual_gif

def evaluate_model(env, model, num_tests=20, max_steps=50):
    """
    Evaluates the model on random trips.
    Reaching **either** of the two goal_nodes counts as success.
    Reports success rate, average reward, distance, steps, detour ratio, etc.
    """
    success_count = 0
    total_rewards = []
    total_distances = []
    total_steps = []
    detour_ratios = []
    step_efficiencies = []
    closer_step_percentages = []
    final_distances_failed = []
    working = False

    for test_idx in range(num_tests):
        state = env.reset()[0]

        if env.current_node in env.goal_nodes:
            print(f"Test {test_idx+1}: Start and goal are the same. Skipping.")
            continue

        visited = [env.current_node]
        cumulative_reward = 0.0
        closer_steps = 0
        done = False
        steps = 0

        start_node = env.current_node
        goal_nodes = env.goal_nodes  
        while not done and steps < max_steps:
            state_tensor = tf.concat([
                tf.one_hot([state[0]], depth=env.state_space, dtype=tf.float32),
                tf.one_hot([state[1]], depth=env.state_space, dtype=tf.float32)
            ], axis=-1)

            q_values = model(state_tensor)
            action = np.argmax(q_values[0].numpy())

            cur_d = min(edistance(env.current_node, g, env.map) for g in goal_nodes)

            next_state, reward, terminated, truncated, _ = env.step(action)

            nxt_d = min(edistance(env.current_node, g, env.map) for g in goal_nodes)
            if nxt_d < cur_d:
                closer_steps += 1

            done = terminated or truncated or (env.current_node in goal_nodes)
            cumulative_reward += reward
            visited.append(env.current_node)
            state = next_state
            steps += 1

        total_distance = 0.0
        for i in range(1, len(visited)):
            x1, y1 = env.map.nodes[visited[i-1]]['x'], env.map.nodes[visited[i-1]]['y']
            x2, y2 = env.map.nodes[visited[i]]['x'], env.map.nodes[visited[i]]['y']
            total_distance += ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        straight_line_distance = min(edistance(start_node, g, env.map) for g in goal_nodes)
        detour_ratio = total_distance / (straight_line_distance + 1e-6)
        step_efficiency = total_distance / (steps + 1e-6)

        if steps > 0:
            percentage_steps_closer = closer_steps / steps
        else:
            percentage_steps_closer = 0.0

        if env.current_node not in goal_nodes:
            final_distance_failed = min(edistance(env.current_node, g, env.map) for g in goal_nodes)
            final_distances_failed.append(final_distance_failed)

        total_rewards.append(cumulative_reward)
        total_distances.append(total_distance)
        total_steps.append(steps)
        detour_ratios.append(detour_ratio)
        step_efficiencies.append(step_efficiency)
        closer_step_percentages.append(percentage_steps_closer)

        if env.current_node in goal_nodes:
            success_count += 1

            if working is False:
                visualize_trip(model, env, visited)
                visual_gif(model, env, visited, gif_path="my_trip.gif")
                working = True
                

        print(f"Test {test_idx+1}: Reward={cumulative_reward:.2f}, Distance={total_distance:.4f}, Steps={steps}, Success={env.current_node in goal_nodes}")

    # Summary
    print("\nEvaluation Summary:")
    print(f"Success Rate: {success_count}/{num_tests} ({(success_count/num_tests)*100:.1f}%)")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Distance Traveled: {np.mean(total_distances):.4f}")
    print(f"Average Steps: {np.mean(total_steps):.1f}")
    print(f"Average Detour Ratio: {np.mean(detour_ratios):.2f}")
    print(f"Average Step Efficiency (Distance per Step): {np.mean(step_efficiencies):.4f}")
    print(f"Average % Steps Moving Closer to Goal: {(np.mean(closer_step_percentages)*100):.1f}%")
    if final_distances_failed:
        print(f"Average Final Distance When Failed: {np.mean(final_distances_failed):.4f}")

    return {
        "success_rate": success_count / num_tests,
        "avg_reward": np.mean(total_rewards),
        "avg_distance": np.mean(total_distances),
        "avg_steps": np.mean(total_steps),
        "avg_detour_ratio": np.mean(detour_ratios),
        "avg_step_efficiency": np.mean(step_efficiencies),
        "avg_closer_steps_pct": np.mean(closer_step_percentages),
        "avg_final_dist_failed": np.mean(final_distances_failed) if final_distances_failed else None
    }
