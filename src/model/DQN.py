import tensorflow as tf 
from tensorflow.keras import layers

# THINGS TO DO: 
    # Make this more complex as needed 
    # Add more models e.g. actor critic in the model folder as needed 

class DQN(tf.keras.Model):
    def __init__(self, state_size, num_actions): 
        self.state_size = state_size
        self.num_actions = num_actions
        
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation = 'relu', input_shape = (state_size,)),
                tf.keras.layers.Dense(32, activation = 'relu'), 
                tf.keras.layers.Dense(num_actions, activation = None)  
            ]
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) 
        self.model.build(input_shape = (None, state_size))
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def call(self, states): 
        
        self.model(states)

    def loss_func(self, batch, discount_factor = 0.99): 

        states, actions, rewards, next_states, done = batch 

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        q_s = self(states)
        q_s_a = tf.reduce_sum(q_s * tf.one_hot(actions, self.num_actions), axis=1)
        q_next_s = self.target_model(next_states)
        q_next_s_a = tf.reduce_max(q_next_s, axis=1)

        loss = tf.reduce_mean(tf.square(q_s_a - (rewards + discount_factor * (1 - done) * q_next_s_a)))

        return loss