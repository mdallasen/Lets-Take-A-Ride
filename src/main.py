import sys
from model.DQN import DQN
from execute.train import train
from execute.inference import visualize_data, visualize_episode
from envs.env_data import oms_data
from envs.env import GraphEnv



# THINGS TO DO: 
    # Complete this when all other functions are done
    # Add in the visualisation from inference here 
    # Add additional models here as needed

def main(): 
    if len(sys.argv) <2 or sys.argv[1] not in {"DQN"}: 
        print("USAGE: python main.py <Model Type>")
        exit()
    if len(sys.argv) == 2: 
        map = oms_data(True)
        env = GraphEnv(map) 
    
    state_size = env.state_space
    print("State Size: ", state_size)
    #state_size = state_size[0]
    num_actions = env.action_space.n
    print('number lllllll',num_actions)

    if sys.argv[1] == "DQN":
        model = DQN(state_size, num_actions)

    totalReward = []
    num_episodes = 650
    memory=None

    for episode in range(num_episodes):
        print(episode, end = "\r")
        if sys.argv[1] == "DQN":
            reward, memory = train(env, model, memory=memory, epsilon = 1-episode/num_episodes)
        else:
            reward = train(env, model)
        if episode in range(0, num_episodes):
            totalReward.append(reward)
    visualize_episode(model, env)
    env.close()
    print(sum(totalReward)/len(totalReward))    
    visualize_data(totalReward)

if __name__ == "__main__": 
    main()