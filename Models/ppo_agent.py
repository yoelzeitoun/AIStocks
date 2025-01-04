import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

from Models.ddpg_agent import OUNoise
from Utilities.shape_manager import ShapeManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, clip_ratio=0.3,
                 update_epochs=10, lam=0.95, tau=0.005, noise_std=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.lam = lam
        self.tau = tau
        self.epsilon = 1.0  # Epsilon-greedy exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Shape Manager
        self.shape_manager = ShapeManager(state_dim, action_dim)

        # Actor and Critic models
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
        self.noise = OUNoise(size=action_dim)

    def build_actor(self):
        """
        Build the actor network.
        """
        state_input = Input(shape=self.state_dim)
        x = Flatten()(state_input)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        out_actions = Dense(self.action_dim, activation='softmax')(x)
        return Model(inputs=state_input, outputs=out_actions)

    def build_critic(self):
        """
        Build the critic network.
        """
        state_input = Input(shape=self.state_dim)
        x = Flatten()(state_input)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        value = Dense(1)(x)
        return Model(inputs=state_input, outputs=value)

    def act(self, state, add_noise=True):
        """
        Generate an action using epsilon-greedy exploration.
        """
        # Ensure correct input shape
        state = self.shape_manager.validate_states(state)

        # Get action probabilities from the actor model
        action_probs = self.actor_model(state).numpy()[0]
        action_probs = np.clip(action_probs, 1e-8, 1 - 1e-8)  # Avoid invalid probabilities
        action_probs /= np.sum(action_probs)  # Ensure sum equals 1

        # Sample an action from the probability distribution
        action = np.random.choice(self.action_dim, p=action_probs)

        # Epsilon-greedy exploration (optional)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)

        # Add noise if needed
        if add_noise:
            noise = self.noise.sample()
            noise = float(noise[0]) if isinstance(noise, (np.ndarray, list)) else float(noise)
            action = np.clip(action + noise, 0, self.action_dim - 1)
            action = int(action)  # Convert to integer after clipping

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return int(action)  # Ensure a single scalar value is returned

    def update_target_networks(self):
        """
        Update target networks using soft updates.
        """
        for target_param, param in zip(self.target_actor_model.trainable_variables, self.actor_model.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic_model.trainable_variables, self.critic_model.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        """
        rewards = np.atleast_1d(rewards)
        values = np.append(values, 0)  # Append 0 for the terminal value
        dones = np.atleast_1d(dones)

        advantages = np.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * last_advantage * (1 - dones[t])

        # Normalize advantages
        advantages_mean = np.mean(advantages)
        advantages_std = np.std(advantages)
        advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        return advantages

    def save_models(self, path="models/ppo"):
        self.actor_model.save_weights(f"{path}_actor.h5")
        self.critic_model.save_weights(f"{path}_critic.h5")

    def load_models(self, path="models/ppo"):
        self.actor_model.load_weights(f"{path}_actor.h5")
        self.critic_model.load_weights(f"{path}_critic.h5")

    def calculate_old_log_probs(self, states, actions):
        """
        Calculate log probabilities for given states and actions.
        """
        logits = self.actor_model(states)
        action_probs = tf.nn.softmax(logits)
        action_probs_taken = tf.reduce_sum(action_probs * tf.one_hot(actions, self.action_dim), axis=1)

        # Avoid log(0) by clipping probabilities
        action_probs_taken = tf.clip_by_value(action_probs_taken, 1e-8, 1.0)
        log_probs = tf.math.log(action_probs_taken)

        # print(f"[DEBUG] - Calculated log_probs: {log_probs[:5].numpy()}")
        return log_probs

    def _apply_gradients(self, grads, optimizer, variables, model_name):
        """
        Apply gradients safely with logging and validation.
        """
        if grads is not None and all(g is not None for g in grads):
            gradient_norm = tf.linalg.global_norm(grads)
            print(f"[DEBUG] - {model_name} gradients norm: {gradient_norm.numpy():.6f}")
            if gradient_norm.numpy() > 1e-8:  # Ensure gradients are valid
                optimizer.apply_gradients(zip(grads, variables))
            else:
                print(f"[ERROR] - {model_name} gradients too small, skipping update!")
        else:
            print(f"[ERROR] - No valid gradients for {model_name} model!")

    def train(self, states, actions, rewards, next_states, dones):
        """
        Train the PPO agent episodically using all collected transitions.
        """
        # Reshape and validate inputs to match critic model's expected shape
        states = tf.convert_to_tensor(states.reshape(-1, self.state_dim[0], self.state_dim[1]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states.reshape(-1, self.state_dim[0], self.state_dim[1]),
                                           dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Validate rewards
        if tf.reduce_any(tf.math.is_nan(rewards)) or tf.reduce_any(tf.math.is_inf(rewards)):
            raise ValueError("Rewards contain NaN or Inf values.")

        # Calculate old log probabilities
        old_log_probs = self.calculate_old_log_probs(states, actions)

        # Compute critic values
        values = tf.squeeze(self.critic_model(states), axis=-1)  # Shape: (batch,)
        next_values = tf.squeeze(self.target_critic_model(next_states), axis=-1)  # Shape: (batch,)

        # Compute returns and advantages
        returns = rewards + self.gamma * next_values * (1 - dones)
        returns = tf.stop_gradient(returns)  # Prevent gradients through returns
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # Train Actor and Critic over multiple epochs
        for _ in range(self.update_epochs):
            # Actor Update
            with tf.GradientTape() as actor_tape:
                logits = self.actor_model(states)
                action_probs = tf.nn.softmax(logits)
                action_probs_taken = tf.reduce_sum(action_probs * tf.one_hot(actions, self.action_dim), axis=1)
                log_probs = tf.math.log(tf.clip_by_value(action_probs_taken, 1e-8, 1.0))
                ratio = tf.exp(log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                surrogate_loss = tf.minimum(ratio * advantages, clipped_ratio * advantages)
                actor_loss = -tf.reduce_mean(surrogate_loss)

            actor_grads = actor_tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self._validate_and_apply_gradients(actor_grads, self.actor_optimizer, self.actor_model.trainable_variables,
                                               "Actor")

            # Critic Update
            with tf.GradientTape() as critic_tape:
                values = tf.squeeze(self.critic_model(states), axis=-1)  # Recompute values for gradient connection
                critic_loss = tf.reduce_mean(tf.square(returns - values))  # MSE loss

            critic_grads = critic_tape.gradient(critic_loss, self.critic_model.trainable_variables)
            self._validate_and_apply_gradients(critic_grads, self.critic_optimizer,
                                               self.critic_model.trainable_variables, "Critic")

        # Update target networks
        self.update_target_networks()

        logger.info(f"Training complete: Actor Loss: {actor_loss.numpy():.4f}, Critic Loss: {critic_loss.numpy():.4f}")

    def _validate_and_apply_gradients(self, grads, optimizer, variables, model_name):
        """
        Validate and apply gradients, ensuring they are not None.
        """
        if grads is None or any(g is None for g in grads):
            logger.error(f"No gradients provided for {model_name}. Check model inputs and loss computation.")
            for var, grad in zip(variables, grads):
                if grad is None:
                    logger.error(f"Gradient for variable {var.name} is None.")
            raise ValueError(f"No gradients provided for {model_name}. Check model inputs and loss computation.")

        # Log gradient norms for debugging
        gradient_norm = tf.linalg.global_norm(grads)
        logger.info(f"{model_name} Gradient Norm: {gradient_norm.numpy():.4f}")

        # Apply gradients
        optimizer.apply_gradients(zip(grads, variables))
