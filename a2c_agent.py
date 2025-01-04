import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam

from Models.ddpg_agent import OUNoise
from Utilities.shape_manager import ShapeManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, noise_std=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.epsilon = 1.0  # Start exploration high
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Initialize shape manager
        self.shape_manager = ShapeManager(state_dim=state_dim, action_dim=action_dim)

        # Build actor and critic networks
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
        self.noise = OUNoise(size=self.action_dim, sigma=noise_std)

    def build_actor(self):
        """
        Build the actor network using LSTM for sequence-based inputs.
        """
        state_input = Input(shape=self.state_dim)
        x = LSTM(64, return_sequences=True)(state_input)
        x = LSTM(32)(x)
        action_probs = Dense(self.action_dim, activation="softmax")(x)
        return Model(inputs=state_input, outputs=action_probs)

    def build_critic(self):
        """
        Build the critic network.
        """
        state_input = Input(shape=self.state_dim)
        flattened_state = Flatten()(state_input)
        x = Dense(64, activation="relu")(flattened_state)
        x = Dense(32, activation="relu")(x)
        value = Dense(1)(x)  # Output: state value
        return Model(inputs=state_input, outputs=value)

    def act(self, state):
        """
        Chooses an action based on the current state using the actor model.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        # Ensure state has the correct shape
        if len(state.shape) > len(self.state_dim) + 1:  # Too many dimensions
            state = state.squeeze(axis=0)  # Remove unnecessary dimensions
        if len(state.shape) == len(self.state_dim):  # Add batch dimension if needed
            state = np.expand_dims(state, axis=0)

        # Predict action probabilities
        action_probs = self.actor_model(state).numpy()[0]  # Get probabilities for batch[0]

        # Debugging: print predicted probabilities
        # print(f"Raw action probabilities: {action_probs}")

        # Clip small values to avoid numerical issues and normalize
        action_probs = np.clip(action_probs, 1e-8, 1.0)  # Prevent extremely small or large values
        action_probs /= action_probs.sum()  # Normalize to ensure sum equals 1

        # Debugging: print normalized probabilities
        # print(f"Normalized action probabilities: {action_probs}")

        # Sample action based on probabilities
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def update_target_networks(self):
        """
        Soft update target networks.
        """
        for target_param, param in zip(self.target_actor_model.trainable_variables,
                                       self.actor_model.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic_model.trainable_variables,
                                       self.critic_model.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def reset_noise(self):
        """
        Reset noise at the beginning of an episode.
        """
        self.noise.reset()

    def train(self, states, actions, rewards, next_states, dones):
        """
        Train the A2C agent using actor-critic loss.
        """
        # Validate inputs
        states = self.shape_manager.validate_states(states)
        next_states = self.shape_manager.validate_states(next_states)
        actions = np.atleast_1d(actions).reshape(-1)
        rewards = np.atleast_1d(rewards)
        dones = np.atleast_1d(dones)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute targets
        next_values = self.target_critic_model(next_states)
        targets = rewards + self.gamma * tf.squeeze(next_values) * (1 - dones)

        # Critic loss
        with tf.GradientTape() as critic_tape:
            values = tf.squeeze(self.critic_model(states))
            critic_loss = tf.reduce_mean((targets - values) ** 2)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        # Actor loss
        with tf.GradientTape() as actor_tape:
            action_probs = self.actor_model(states, training=True)
            action_mask = tf.one_hot(actions, self.action_dim, dtype=tf.float32)
            log_probs = tf.reduce_sum(action_mask * tf.math.log(action_probs + 1e-8), axis=1)
            advantages = targets - values
            actor_loss = -tf.reduce_mean(log_probs * advantages)

        actor_grads = actor_tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Update target networks
        self.update_target_networks()

        # Debugging
        print(f"A2C Actor loss: {actor_loss.numpy():.6f}, Critic loss: {critic_loss.numpy():.6f}")
