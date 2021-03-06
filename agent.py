from keras import layers, models, optimizers
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
from collections import namedtuple, deque
import math
from ounoise import OUNoise


class DeepQAgent:
    """
        Deep Q Learning based agent customized for Balloma Video Game domain
        A Deep Network is the Q-Value function.
        Action space equals to 359 (i.e possible angles of the ball moving vector)
    """


    def __init__(self, env):
        """Initialize parameters and build model.
            Params
            ======
            state_size (int): Frames set shape
        """
        self.session = K.get_session()
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.env = env
        self.state_size = self.env.state_size
        self.score = -math.inf
        self.best_score = -math.inf
        self.last_loss = math.inf
        self.buffer_size = 100000
        self.batch_size = 16
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.gamma = 0.99
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.0001
        self.learning_rate = 0.0001
        self.vector_size = 100
        self.delay = 500
        self.current_step = 0
        self.build_model()

    def reset_episode(self):
        self.total_reward = 0
        state = self.env.reset()
        self.last_state = state
        return state

    def build_model(self):
        """Build a network that represents Q-Values states -> Q-Values."""
        # Define input layer (states)
        states = layers.Input(shape=self.state_size, name='states')

        net = layers.Lambda(lambda x: x/255)(states)

        # Add hidden layers
        net = layers.Conv2D(filters=32, kernel_size=2,
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            activation='relu')(states)

        net = layers.Conv2D(filters=32, kernel_size=2,
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            activation='relu')(net)

        net = layers.Conv2D(filters=32, kernel_size=2,
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            activation='relu')(net)


        net = layers.Dense(units=200, activation='relu')(net)
        net = layers.Dense(units=200, activation='relu')(net)

        net = layers.Dense(units=359, activation='sigmoid',
                                   name='Q_values')(net)

        Q_values = layers.GlobalAveragePooling2D()(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=Q_values)
        optimizer = optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.summary()

    def step(self, action, reward, next_state, done):
        action = action[1]
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward

        self.last_state = next_state
        self.current_step += 1

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""

        Q_values = self.model.predict(np.array([state]))[0]

        # Whether to explore or exploit
        explore_p = self.explore_stop + (self.explore_start
                                    - self.explore_stop)*np.exp(-self.decay_rate*self.current_step)

        angle = int(np.random.uniform() * 359)
        if explore_p < np.random.rand():
            angle = np.argmax(Q_values)

        return np.array([self.vector_size, angle, self.delay])


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not
                            None])
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.array([e.next_state for e in experiences if e is not None])
        self.score = rewards.mean()
        self.best_score = max(self.score, self.best_score)

        Q_targets_next = self.model.predict_on_batch([next_states])
        Q_targets = rewards + self.gamma * np.max(Q_targets_next, axis=1)


        Q_values = self.model.predict_on_batch([states])

        for i in range(states.shape[0]):
            for j in range(actions.shape[0]):
                ai = actions[j]
                Q_values[i, ai] = Q_targets[i, j]

        d = self.model.fit(states, Q_values)
        self.last_loss = d.history['loss'][0]


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
            Params
            ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=self.state_size, name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Flatten()(states)
        net_states = layers.Dense(units=32, activation='relu',
                                  kernel_regularizer=l2(0.01),
                                  bias_regularizer=l2(0.01))(net_states)

        net_states = layers.Dense(units=64, activation='relu',
                                  kernel_regularizer=l2(0.01),
                                  bias_regularizer=l2(0.01))(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu',
                                  kernel_regularizer=l2(0.01),
                                  bias_regularizer=l2(0.01))(actions)

        net_actions = layers.Dense(units=64, activation='relu',
                                  kernel_regularizer=l2(0.01),
                                  bias_regularizer=l2(0.01))(net_actions)


        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values',
                                kernel_initializer=RandomUniform(minval=-0.0003,
                                                                 maxval=0.0003),
                                bias_initializer=RandomUniform(minval=-0.0003,
                                                                 maxval=0.0003)
                                )(net)
                                                                                                                                                                                                                     # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.
            Params
            ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=self.state_size, name='states')

        net = layers.Lambda(lambda x: x/255)(states)

        # Add hidden layers
        net = layers.Conv2D(filters=32, kernel_size=2,
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            activation='relu')(states)

        net = layers.Conv2D(filters=32, kernel_size=2,
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            activation='relu')(net)

        net = layers.Conv2D(filters=32, kernel_size=2,
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            activation='relu')(net)


        net = layers.Dense(units=200, activation='relu')(net)
        net = layers.Dense(units=200, activation='relu')(net)
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                   name='raw_actions',
                                   kernel_initializer=RandomUniform(minval=-0.0003,
                                                                 maxval=0.0003),
                                bias_initializer=RandomUniform(minval=-0.0003,
                                                                 maxval=0.0003))(net)

        raw_actions = layers.GlobalAveragePooling2D()(raw_actions)
        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)
        # Define optimizer and training function
        optimizer = optimizers.Adam(learning_rate=0.001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients,
                                           K.learning_phase()],
                                   outputs=[self.model.output],
                                   updates=updates_op)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        print(f"Memory next state shape: {next_state.shape}")
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.session = K.get_session()
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.score = -math.inf
        self.best_score = -math.inf
        self.last_loss = math.inf

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu,
                             self.exploration_theta, self.exploration_sigma)

        self.noise_scale = (self.exploration_mu, self.exploration_theta,
                            self.exploration_sigma)
        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 16
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.001 # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        self.total_reward = 0
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward

        # Learn, if enough samples are available in memory
        print("Memory Size: {}, Batch Size: {}".format(len(self.memory),
                                                       self.batch_size))
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        #state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(np.array([state]))[0]
        return list(action + self.noise.sample()) # add some noise for exploration

    def learn(self, experiences):
        print("Fitting model iteration ...")
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.array([e.next_state for e in experiences if e is not None])
        print("Next states shape: {}".format(next_states.shape))
        self.score = rewards.mean()
        self.best_score = max(self.score, self.best_score)

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        r = self.actor_local.train_fn([states, action_gradients, 1])

        self.last_loss = np.mean(-action_gradients * actions)

        # custom training function Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


    if __name__ == "__main__":
        state_size = (84, 296, 9)
        action_low = np.array([1, 0 ,1])
        action_high = np.array([10, 359, 2000])
        net = Actor(state_size, 3, action_low, action_high)
        #net = Critic(state_size, 3)
        net.model.summary()
