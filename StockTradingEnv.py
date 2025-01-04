import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box
from Data.data import Data
# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class StockTradingEnv(Env):
    """A stock trading environment for gym"""

    def __init__(self, data, scaler, lookback_window_size=5, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.scaler = scaler
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance

        # Action and observation space definitions
        self.action_space = Discrete(3)  # Sell, Hold, Buy
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(lookback_window_size, data.shape[1]), dtype=np.float32
        )

        # Initialize state
        self.portfolio_value = initial_balance  # Track portfolio value
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.done = False
        self.episode_rewards = []
        self.current_price = self.inverse_transform_price(self.data[self.current_step, -1])

        # Debug statement to ensure price is correct
        if self.current_price == 0:
            print("[ERROR] Initial price is zero! This should not happen.")

        return self.data[self.current_step:self.current_step + self.lookback_window_size]

    def inverse_transform_price(self, scaled_price):
        # Reshape price for scaler compatibility
        actual_price = self.scaler.inverse_transform([[scaled_price]])[0][0]
        return actual_price

    def reset_with_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")
        if len(data.shape) != 2 or data.shape[1] != self.data.shape[1]:
            raise ValueError(f"Data shape mismatch. Expected (n_samples, {self.data.shape[1]}), got {data.shape}")

        self.data = data
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.shares_held = 0
        self.episode_rewards = []

        # Ensure no NaN or zero in prices (if using prices)
        if np.isnan(self.data).any():
            raise ValueError("Data passed to environment contains NaN values!")
        if (self.data == 0).any():
            raise ValueError("Data passed to environment contains zero values!")

        self.state = self.data[self.current_step:self.current_step + self.lookback_window_size]
        return self.state

    def step(self, action):
        """
        Execute one step in the environment based on the chosen action.
        """
        logger.debug(f"Action received in step(): {action}, shape: {np.shape(action)}")

        # Validate action
        if isinstance(action, np.ndarray) and action.size > 1:
            raise ValueError(f"Action must contain a single value, but got shape {action.shape} with values {action}.")
        action = np.squeeze(action).item()  # Convert to scalar integer

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be 0 (Sell), 1 (Hold), or 2 (Buy).")

        # Store previous portfolio value for reward calculation
        previous_portfolio_value = self._calculate_portfolio_value()

        # Move to the next step
        self.current_step += 1
        if self.current_step + self.lookback_window_size >= len(self.data):
            self.done = True
        else:
            self.current_price = self.inverse_transform_price(self.data[self.current_step, -1])

        print(f"\n--- Step {self.current_step} ---")
        print(f"Current Price: {self.current_price:.2f}")
        print(f"Action: {action}, Shares Held: {self.shares_held}, Balance: {self.balance:.2f}")

        # Execute action: 0 = Sell (Short or Sell Long), 1 = Hold, 2 = Buy
        if action == 0:  # SELL
            if self.shares_held > 0:  # Sell all long shares
                self.balance += self.shares_held * self.current_price
                print(f"Sold all long shares. New Balance: {self.balance:.2f}")
                self.shares_held = 0
            elif self.balance >= self.current_price:  # Short 1 share
                self.balance -= self.current_price  # Subtract current price for short
                self.shares_held -= 1
                print(f"Shorted 1 share. New Shares Held: {self.shares_held}, Balance: {self.balance:.2f}")
            else:
                print("Not enough balance to short a share.")

        elif action == 2:  # BUY
            if self.shares_held < 0:  # Cover short positions
                self.balance += abs(self.shares_held) * self.current_price  # Pay back the price to cover short
                self.shares_held = 0
                print(f"Covered all short shares. New Balance: {self.balance:.2f}")
            elif self.balance >= self.current_price:  # Buy 1 long share
                self.balance -= self.current_price
                self.shares_held += 1
                print(f"Bought 1 long share. Shares Held: {self.shares_held}, Balance: {self.balance:.2f}")
            else:
                print("Not enough balance to buy a long share.")

        # End of Episode: Sell all remaining shares
        if self.done:
            if self.shares_held > 0:  # Sell all long shares
                self.balance += self.shares_held * self.current_price
                print(f"[End of Episode] Sold all long shares: {self.shares_held}")
            elif self.shares_held < 0:  # Cover all short shares
                self.balance += self.shares_held * self.current_price  # Adjust for short positions
                print(f"[End of Episode] Covered all short shares: {self.shares_held}")
            self.shares_held = 0

        # Update portfolio value using the new logic
        self.portfolio_value = self._calculate_portfolio_value()

        # Print the current state for debugging
        print(
            f"Shares Held: {self.shares_held}, Balance: {self.balance:.2f}, Portfolio Value: {self.portfolio_value:.2f}")

        # Calculate reward as change in portfolio value
        reward = self.portfolio_value - previous_portfolio_value
        print(f"Reward: {reward:.2f}")

        # Define next state
        next_state = (
            self.data[self.current_step:self.current_step + self.lookback_window_size]
            if not self.done
            else np.zeros((self.lookback_window_size, self.data.shape[1]))
        )

        return next_state, reward, self.done, {}

    def _calculate_portfolio_value(self):
        """
        Calculate the real portfolio value as:
        Cash Balance + (Absolute Value of Shares Held * Current Price).
        """
        portfolio_value = self.balance + (abs(self.shares_held) * self.current_price)
        return portfolio_value

    def render(self, mode='human', close=False):
        current_price = self.current_price
        profit = self.balance + self.shares_held * current_price - self.initial_balance

        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares Held: {self.shares_held}")
        print(f"Current Price: {current_price}")
        print(f"Total Profit: {profit}")

    def get_episode_rewards(self):
        return self.episode_rewards
