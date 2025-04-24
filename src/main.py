import sys
from model.DQN import DQN
from execute.train import train
from execute.inference import visualize_data, visualize_episode
from envs.env_data import oms_data
from envs.env import GraphEnv

def main(): 
    map = oms_data()
    env = GraphEnv(map)
    state_size = len(env.nodes)

    num_actions = env.action_space.n
    model = DQN(state_size, num_actions)

    totalReward = []
    num_episodes = 10
    memory=None

    for episode in range(num_episodes):
        print(episode, end = "\r")
        reward, memory = train(env, model, memory=memory, epsilon = 1-episode/num_episodes)

        if episode in range(0, num_episodes):
            totalReward.append(reward)
        env.close()

    print(sum(totalReward)/len(totalReward))  

if __name__ == "__main__": 
    main()