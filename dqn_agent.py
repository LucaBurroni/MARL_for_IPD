import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # suppress TensorFlow info messages
from tensorflow import keras
from collections import deque

class DQNAgent:
    def __init__(self, input_shape, n_outputs):
        
        # input_shape and n_outputs passed as parameters
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        
        # build both model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # replay buffer
        self.replay_buffer = deque(maxlen=2000)
        
        # define optimizer and loss function
        self.optimizer=keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn=keras.losses.mean_squared_error
        
        # keep track of loss history to plot it after training
        self.loss_history = []

    def build_model(self):
        model = keras.Sequential([
            # two hidden layers with 32 units
            keras.layers.Dense(32, activation="relu", input_shape=self.input_shape),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(self.n_outputs)
        ])
        return model

    def select_action(self, state, epsilon):
        # define epsilon greedy policy for selecting actions
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose = 0)
            return np.argmax(Q_values[0])

    def store_experience(self, state, action, reward, next_state, done):
        # store experience in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self, batch_size):
        # sample experiences from replay buffer
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def train(self, batch_size, gamma=0.95):
        # check that replay buffer has enough experiences
        if len(self.replay_buffer) < batch_size:
            return

        # model train step
        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)

        next_Q_values = self.target_model.predict(next_states, verbose = 0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - dones) * gamma * max_next_Q_values

        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

         # record loss in history list
        self.loss_history.append(loss.numpy())


    def update_target_network(self):
        # update target network
        self.target_model.set_weights(self.model.get_weights())