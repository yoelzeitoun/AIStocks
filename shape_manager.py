import numpy as np

class ShapeManager:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def validate_data_shape(self, data, expected_features, name):
        """
        Validate the shape of the data.
        Args:
            data (np.ndarray): The data to validate.
            expected_features (int): The expected number of features.
            name (str): The name of the dataset for logging.
        Raises:
            ValueError: If the shape is invalid.
        """
        if data.shape[-1] != expected_features:
            raise ValueError(f"{name} features mismatch! Expected {expected_features}, got {data.shape[-1]}")

    def validate_states(self, states):
        """
        Validate and reshape states to match the expected dimensions.
        """
        if len(states.shape) == len(self.state_dim):  # Missing batch dimension
            states = np.expand_dims(states, axis=0)
        return states

    def validate_actions(self, actions, batch_size):
        """
        Validate and reshape actions to ensure consistency with action_dim and batch_size.
        """
        if len(actions.shape) == 1:  # Single action, reshape to (batch_size, action_dim)
            actions = actions.reshape(batch_size, self.action_dim)
        elif actions.shape[0] != batch_size or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch! Expected ({batch_size}, {self.action_dim}), got {actions.shape}")
        return actions

    def validate_rewards(self, rewards, batch_size):
        """
        Validate and reshape rewards to match the batch size.
        """
        rewards = np.atleast_1d(rewards).reshape(-1)
        if rewards.shape[0] != batch_size:
            raise ValueError(f"Rewards batch size mismatch! Expected {batch_size}, got {rewards.shape[0]}")
        return rewards

    def validate_dones(self, dones, batch_size):
        """
        Validate and reshape done flags to match the batch size.
        """
        dones = np.atleast_1d(dones).reshape(-1)
        if dones.shape[0] != batch_size:
            raise ValueError(f"Dones batch size mismatch! Expected {batch_size}, got {dones.shape[0]}")
        return dones
