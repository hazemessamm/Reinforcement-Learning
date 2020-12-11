import gym
import numpy as np
import tensorflow as tf
from itertools import count
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from collections import deque
import random


class ExperienceReplay:
    #this class is for storing information like current state, next state, action and reward
    #this info will help us in training our neural network
    #the advantage of this is that will make our neural network have better convergence
    def __init__(self, batch_size=1024, max_size=4096):
        assert batch_size <= max_size
        self.max_size = max_size
        #our buffer will be double ended queue because we don't need to worry about the max size.
        #once a new experience is added it will remove the oldest one
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
    
    def add_experience(self, experience):
        'takes a tuple of (current state, next state, action, reward, done)'
        #we append every new experience we get to the queue
        self.buffer.append(experience)
    
    def sample(self, preprocessed=True):
        'gets a sample from the queue and preprocess it if "preprocessed" is true'
        #Samples the batch from the queue with the specified batch size.
        #I used random.sample instead of random.choices because I don't want any example to be sampled more than once
        batch = random.sample(self.buffer, k=self.batch_size)
        #if processed is true it will preprocess the batch in order to return each column instead of returning batch as it is 
        if preprocessed:
            return self.preprocess_batch(batch)
        else:
            return batch
    
    def preprocess_batch(self, batch):
        'converts every column into tf.tensor and return a tuple of columns'
        current_states = tf.convert_to_tensor([x[0] for x in batch], dtype=tf.float32) 
        next_states = tf.convert_to_tensor([x[1] for x in batch], dtype=tf.float32)
        action_probs = tf.convert_to_tensor([x[2] for x in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([x[3] for x in batch], dtype=tf.float32)
        rewards = tf.convert_to_tensor([x[4] for x in batch], dtype=tf.float32)
        done = tf.convert_to_tensor([x[5] for x in batch], dtype=tf.float32)
        
        return current_states, next_states, action_probs, actions, rewards, done


class Agent:
    def __init__(self, action_space, state_space, buffer, discount_factor=0.99):
        self.action_space = action_space
        self.state_space = state_space
        self.network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.network.get_weights())
        self.target_network.trainable = False
        self.network_optimizer = tf.keras.optimizers.Adam()
        self.discount_factor = discount_factor
        self.buffer = buffer
        self.update_counter = 0
        self.epsilon = 0.1
        self.network_loss = tf.keras.losses.MeanSquaredError()

    def create_network(self):
        inputs = keras.Input(shape=(self.state_space,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_space, activation='softmax')(x)
        return Model(inputs, outputs)

    def fit(self, epochs=1):
        if len(self.buffer.buffer) < self.buffer.batch_size:
            return

        for _ in range(epochs):
            states, next_states, _, _, rewards, _ = self.buffer.sample()
            next_action_probs = self.target_network(next_states)
            td_target = rewards + self.discount_factor * tf.reduce_max(next_action_probs, axis=-1)
            td_target = tf.reshape(td_target, (-1, 1))
            self.__train_step(states, rewards, td_target)

        if self.update_counter % 15 == 0:
            self.target_network.set_weights(self.network.get_weights())

        self.update_counter += 1

    def __train_step(self, states, rewards, td_target):
        with tf.GradientTape() as tape:
            action_probs = self.network(states)
            td_error = self.network_loss(td_target, action_probs)
            grads = tape.gradient(td_error, self.network.trainable_weights)
        
        self.network_optimizer.apply_gradients(zip(grads, self.network.trainable_weights))

    def policy(self, state):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        action_probs = np.squeeze(self.network(state))

        actions = np.ones(self.action_space, dtype=np.float32) * self.epsilon / self.action_space
        best_action = np.argmax(action_probs)
        actions[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(self.action_space, p=actions)
        
        return int(action), actions


class Environment:
    def __init__(self, num_episodes=1000, render_env=False):
        self.env = gym.make('MountainCar-v0')
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.buffer = ExperienceReplay()
        self.agent = Agent(self.action_space, self.state_space, self.buffer)
        self.num_episodes = num_episodes
        self.stats = {'total_rewards': [], 'timesteps_per_episode': []}
        self.render_env = render_env

    def run(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            total_timesteps = 0
            t = 0
            while True:
                if self.render_env:
                    self.env.render()
                action, action_probs = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add_experience((state, next_state, action_probs, action, reward, done))
                t += 1
                total_reward += reward
                total_timesteps += t
                if done:
                    self.stats['total_rewards'].append(total_reward)
                    self.stats['timesteps_per_episode'].append(total_timesteps)
                    print(f'Episode: {episode}/{self.num_episodes}, Total reward: {total_reward}, Total timesteps: {total_timesteps}')
                    self.agent.fit(5)
                    break
                
                state = next_state


env = Environment(render_env=True)
env.run()