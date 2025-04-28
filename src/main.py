from model.DQN import DQN
from execute.train import train
from execute.inference import visualize_data, visualize_trip, visual_gif
from envs.env_data import oms_data
from envs.env import GraphEnv

def main(): 
    map = oms_data(True)
    env = GraphEnv(map)
    state_size = len(env.nodes)

    num_actions = env.action_space.n
    model = DQN(state_size, num_actions)

    totalReward = []
    num_episodes = 50
    memory=None

    for episode in range(num_episodes):
        print(episode, end = "\r")
        reward, memory = train(env, model, memory=memory, epsilon = 1-episode/num_episodes)

        if episode in range(0, num_episodes):
            totalReward.append(reward)
        env.close()

    print(f"\nAverage Reward over {num_episodes} episodes: {sum(totalReward)/len(totalReward):.2f}")

    visualize_trip(model, env)
    visual_gif(model, env, gif_path="my_trip.gif")
    visualize_data(totalReward)

if __name__ == "__main__": 
    main()