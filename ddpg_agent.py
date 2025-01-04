import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

from Utilities.shape_manager import ShapeManager
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

        # Debugging
        # print(f"OUNoise initialized with size={size}, mu={mu}, theta={theta}, sigma={sigma}")

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal(size=self.size)
        self.state += dx
        return self.state

    def reset(self):
        self.state = np.ones(self.size) * self.mu


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def check(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, noise_std=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Initialize shape manager
        self.shape_manager = ShapeManager(state_dim=state_dim, action_dim=action_dim)

        # Initialize actor and critic networks
        self.actor_model = self.build_actor()
        self.target_actor_model = self.build_actor()
        self.target_actor_model.set_weights(self.actor_model.get_weights())

        self.critic_model = self.build_critic()
        self.target_critic_model = self.build_critic()
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        # Optimizers
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

        # Noise for exploration
        self.noise = OUNoise(size=self.action_dim)

    def build_actor(self):
        """
        Build the actor network.
        """
        state_input = Input(shape=self.state_dim)  # Input matches state_dim
        x = Flatten()(state_input)  # Flatten the input from (5, 19) to (5*19,)
        x = Dense(32, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        action = Dense(self.action_dim, activation="tanh")(x)  # Output matches action_dim
        return Model(state_input, action)

    def build_critic(self):
        """
        Build the critic network.
        """
        state_input = Input(shape=self.state_dim)
        action_input = Input(shape=(self.action_dim,))

        # Flatten and concatenate inputs
        flattened_state = Flatten()(state_input)
        x = Concatenate(axis=-1)([flattened_state, action_input])

        x = Dense(32, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        value = Dense(1)(x)  # Output: Q-value

        return Model([state_input, action_input], value)

    def act(self, state, add_noise=True):
        """
        Generate an action based on the current state or batch of states using an epsilon-greedy strategy.
        """
        state = self.shape_manager.validate_states(state)

        # Reshape state to match actor input
        if len(state.shape) == 2:  # If state is 2D, reshape to 3D
            state = state.reshape(-1, *self.state_dim)

        logger.debug(f"Input state shape for actor model: {state.shape}, expected: {self.state_dim}")

        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            action = np.random.uniform(-1, 1, self.action_dim)  # Random action for exploration
        else:
            action = self.actor_model(state).numpy()

        logger.debug(f"Actor model output action shape: {action.shape if isinstance(action, np.ndarray) else 'N/A'}, expected action_dim: {self.action_dim}")

        # Add noise for continuous exploration
        if add_noise:
            noise = self.noise.sample()
            action += noise

        action = (action + 1) / 2  # Rescale action to [0, 1]
        action = np.clip(action, 0, 1)  # Ensure action bounds are respected

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # logger.info(f"Updated epsilon to {self.epsilon:.4f}")

        return action

    def update_target_networks(self):
        """
        Update target networks using soft updates.
        """
        for target_param, param in zip(self.target_actor_model.trainable_variables, self.actor_model.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic_model.trainable_variables, self.critic_model.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def save_models(self, path="models/ddpg"):
        self.actor_model.save_weights(f"{path}_actor.h5")
        self.critic_model.save_weights(f"{path}_critic.h5")

    def load_models(self, path="models/ddpg"):
        self.actor_model.load_weights(f"{path}_actor.h5")
        self.critic_model.load_weights(f"{path}_critic.h5")

    def reset_noise(self):
        """
        Reset the noise process at the beginning of each episode.
        """
        self.noise.reset()

    def train(self, states, actions, rewards, next_states, dones):
        """
        Train the DDPG agent.
        """
        # Validate shapes
        states = self.shape_manager.validate_states(states)
        next_states = self.shape_manager.validate_states(next_states)
        actions = self.shape_manager.validate_actions(actions, batch_size=states.shape[0])
        rewards = self.shape_manager.validate_rewards(rewards, batch_size=states.shape[0])
        dones = self.shape_manager.validate_dones(dones, batch_size=states.shape[0])

        # Convert inputs to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Ensure `actions` matches the batch size of `states`
        if actions.shape[0] != states.shape[0]:
            actions = tf.repeat(actions, repeats=states.shape[0], axis=0)

        # Add assertion here to validate batch sizes
        assert states.shape[0] == actions.shape[0], f"Batch size mismatch: states {states.shape}, actions {actions.shape}"

        # Update critic
        target_actions = self.target_actor_model(next_states)
        if len(target_actions.shape) == 3:  # Handle sequence output
            target_actions = target_actions[:, -1, :]  # Use the last time step

        target_q_values = self.target_critic_model([next_states, target_actions])
        target_q_values = rewards + self.gamma * (1 - dones) * tf.squeeze(target_q_values, axis=1)

        with tf.GradientTape() as tape:
            q_values = tf.squeeze(self.critic_model([states, actions]), axis=1)
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            if len(actions_pred.shape) == 3:  # Handle sequence output
                actions_pred = actions_pred[:, -1, :]  # Use the last time step

            actor_loss = -tf.reduce_mean(self.critic_model([states, actions_pred]))

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Update target networks
        self.update_target_networks()

        # Debugging for losses
        # print(f"DDPG Actor loss: {actor_loss.numpy()}, Critic loss: {critic_loss.numpy()}")
