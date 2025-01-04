from Data.data import Data, DataPreprocessor
from Models.ppo_agent import PPOAgent
from Models.ensemble_agent import EnsembleAgent
from Evaluation.plotter import PredictionVisualizer
import numpy as np
from Simulation.StockTradingEnv import StockTradingEnv
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize data
data_handler = Data('GOLD', RSI=False, MA=False, BB=False, PP=False, FIB=False, drop_ohl=True, lookback=5)
data_preprocessor = DataPreprocessor(data_handler.prices, lookback=5)
x_train, y_train = data_preprocessor.x_train, data_preprocessor.y_train
x_test, y_test = data_preprocessor.x_test, data_preprocessor.y_test

# Check for NaN in the data
datasets = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
for name, dataset in datasets.items():
    if np.isnan(dataset).any():
        logger.error(f"{name} contains NaN values at indices: {np.argwhere(np.isnan(dataset))}")
        raise ValueError(f"{name} contains NaN values! Please clean your data. Shape: {dataset.shape}")

# Initialize PPO agent
state_dim = x_train.shape[1:]
logger.info(f"State dimensions set to: {state_dim}")
action_dim = 3  # Define based on buy, hold, sell actions
ppo_agent = PPOAgent(state_dim, action_dim)

# Ensemble Agent setup
agents = {'PPO': ppo_agent}
ensemble_agent = EnsembleAgent(agents)

# Manually select the PPO agent as the best agent
ensemble_agent.current_agent = ppo_agent  # Explicitly set the PPO agent
logger.info("PPO Agent has been selected as the current agent.")

# Reshape data for the environment
x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
x_train_reshaped[x_train_reshaped == 0] = 0.01  # Avoid zero values
x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])

# Initialize Train Environment
env = StockTradingEnv(x_train_reshaped, data_preprocessor.scaler)

# Track rewards and portfolio values for train phase
train_rewards = []
train_portfolio_values = [10000]  # Initial portfolio value
train_prices = []

# Training Phase
num_episodes = 2  # Adjust as needed
try:
    for episode in range(num_episodes):
        logger.info(f"\nStarting Episode {episode + 1}/{num_episodes}")
        state = env.reset_with_data(x_train_reshaped)
        done = False
        current_balance = 10000
        shares_held = 0
        episode_rewards = []

        while not done:
            # Terminate loop if price data is exhausted
            if env.current_step + env.lookback_window_size >= len(env.data):
                logger.warning("Price data exhausted. Ending the current episode.")
                done = True
                break

            # Reshape state if necessary
            if len(state.shape) == 2:
                state = state.reshape(1, *state.shape)

            # Perform action and step environment
            action = ensemble_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            current_price = env.inverse_transform_price(env.data[env.current_step - 1, -1])

            # Update portfolio value
            current_balance += reward
            portfolio_value = current_balance + (shares_held * current_price)

            # Collect rewards and portfolio values
            episode_rewards.append(reward)
            train_portfolio_values.append(portfolio_value)
            train_prices.append(current_price)

            state = next_state

        # Store rewards for this episode
        train_rewards.extend(episode_rewards)
        logger.info(f"Episode {episode + 1} completed with Total Reward: {sum(episode_rewards):.2f}")

except Exception as e:
    logger.error(f"Simulation failed: {e}")
    raise

# Test Phase
logger.info("\nStarting Test Phase:")
test_env = StockTradingEnv(x_test_reshaped, data_preprocessor.scaler)
state = test_env.reset()
test_rewards = []
test_portfolio_values = [10000]  # Initial portfolio value
test_prices = []
current_balance = 10000
shares_held = 0

done = False
while not done:
    # Ensure correct state shape
    state = state.reshape(1, *state.shape)
    action = ensemble_agent.act(state)
    next_state, reward, done, _ = test_env.step(action)
    current_price = test_env.inverse_transform_price(test_env.data[test_env.current_step - 1, -1])

    # Update portfolio value
    current_balance += reward
    portfolio_value = current_balance + (shares_held * current_price)

    # Collect rewards and portfolio values
    test_rewards.append(reward)
    test_portfolio_values.append(portfolio_value)
    test_prices.append(current_price)

    state = next_state

logger.info(f"Test Phase Completed. Total Test Rewards: {sum(test_rewards):.2f}")

# Initialize PredictionVisualizer for Train and Test
prediction_visualizer = PredictionVisualizer(
    model=ppo_agent.actor_model,
    data=data_preprocessor,
    rewards={"Train": train_rewards, "Test": test_rewards},
    portfolio_values={"Train": train_portfolio_values, "Test": test_portfolio_values},
    reward_timestamps=[],
    initial_portfolio=10000,
    prices={"Train": train_prices, "Test": test_prices},
    plot=True
)
