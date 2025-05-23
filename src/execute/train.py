import numpy as np
import tensorflow as tf
from model.DQN import DQN

def train_episode(env, model, batch_size, memory, epsilon=.1):

    # train_model for one episode
    state = env.reset()[0]
    done = False
    ep_rwd = []
    num_batches = 10

    # e greedy approach to selecting action
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else: 
            state_vector = tf.concat([
                tf.one_hot([state[0]], depth=env.state_space, dtype=tf.float32),
                tf.one_hot([state[1]], depth=env.state_space, dtype=tf.float32)
            ], axis=-1)
            q_values = model(state_vector)
            action = np.argmax(q_values[0].numpy())

        # determine the rewards, next state etc. from that action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_rwd.append(reward)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # for each data point in the batch, compute the states, actions and rewards and determine the loss, then back prop
        if len(memory) > batch_size:
            for _ in range(num_batches):
                indices = np.random.randint(0, len(memory), size=batch_size)
                batch = [memory[i] for i in indices]
                states, actions, rewards, next_states, dones = zip(*batch)
                                
                states = [
                    tf.concat([
                        tf.one_hot([state[0]], depth=env.state_space, dtype=tf.float32),
                        tf.one_hot([state[1]], depth=env.state_space, dtype=tf.float32)
                    ], axis=-1) 
                    for state in states
                ]

                next_states = [
                    (next_state[0], next_state[1]) if isinstance(next_state, (tuple, list)) else (next_state, next_state)
                    for next_state in next_states
                ]
                
                next_states = [
                    tf.concat([
                        tf.one_hot([next_state[0]], depth=env.state_space, dtype=tf.float32),
                        tf.one_hot([next_state[1]], depth=env.state_space, dtype=tf.float32)
                    ], axis=-1) 
                    for next_state in next_states
                ]

                states = tf.squeeze(states) 
                next_states = tf.squeeze(next_states) 
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                dones = tf.convert_to_tensor(dones, dtype=tf.bool)

                with tf.GradientTape() as tape: 
                    loss = model.loss_func((states, actions, rewards, next_states, dones))

                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            model.target_model.set_weights(model.model.get_weights())

    return sum(ep_rwd), memory

def train(env, model, memory = None, epsilon=.1): 

    if isinstance(model, DQN): 

        if memory is None:
            memory = []
            while len(memory) < 1:
                state = env.reset()[0]
                ep_memory = []
                for i in range(50): 
                    action = np.random.randint(0, model.num_actions)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    memory.append((state, action, reward, next_state, done))
                    state = next_state
                    if done: 
                        break
                memory.extend(ep_memory)

        memory = memory[-500:]
        total_r, memory = train_episode(env, model, batch_size=5, memory=memory, epsilon=epsilon)
        
        return total_r, memory

    else:
        raise ValueError("Unsupported model type.")