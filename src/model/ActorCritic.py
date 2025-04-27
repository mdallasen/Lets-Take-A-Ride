import tensorflow as tf
from tensorflow.keras import layers

class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions

    
        self.actor = tf.keras.Sequential([
            layers.Input(shape=(state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_actions, activation='softmax')
        ])
       
        self.critic = tf.keras.Sequential([
            layers.Input(shape=(state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  
        ])

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, state):
        """Returns action probabilities and value estimate"""
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

    def train_step(self, state, action, reward, next_state, done, gamma=0.99):
        """
        Performs one actor-critic training step.
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.int32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        done = tf.convert_to_tensor([done], dtype=tf.float32)

    
        _, next_value = self(next_state)
        target = reward + gamma * (1 - done) * tf.squeeze(next_value)

        with tf.GradientTape() as tape:
            _, value = self(state)
            value = tf.squeeze(value)
            critic_loss = tf.square(target - value)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            probs, _ = self(state)
            action_prob = tf.gather(tf.squeeze(probs), action)
            advantage = target - value
            actor_loss = -tf.math.log(action_prob + 1e-10) * tf.stop_gradient(advantage)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy()
