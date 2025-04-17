import tensorflow as tf 
from tensorflow.keras import layers

# THINGS TO DO: 
    # Make this more complex

class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size): 
        self.state_size = state_size
        self.dense1 = layers.Dense(128, activation = 'relu')
        self.dense2 = layers.Dense(64, activation = 'relu')
        self.output = layers.Dense(action_size, activation = 'linear')

    def call(self, inputs): 
        x = tf.one_hot(inputs, depth = self.state_size)
        x = tf.cast(x, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)