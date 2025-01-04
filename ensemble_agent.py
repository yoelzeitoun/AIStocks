import numpy as np
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleAgent:
    def __init__(self, agents, evaluation_window=30):
        self.agents = agents  # A dictionary: {'PPO': PPOAgent, 'A2C': A2CAgent, 'DDPG': DDPGAgent}
        self.evaluation_window = evaluation_window
        self.current_agent = None
        self.current_agent_name = None  # Track the name of the current agent

    def get_current_model(self):
        """
        Returns the prediction model of the currently selected agent.
        """
        if self.current_agent is None:
            raise Exception("No agent selected. Please call select_best_agent first.")
        if hasattr(self.current_agent, 'actor_model'):
            return self.current_agent.actor_model
        elif hasattr(self.current_agent, 'model'):  # Generic fallback
            return self.current_agent.model
        else:
            raise Exception("The selected agent does not have a valid prediction model.")

    def select_best_agent(self, sharpe_ratios):
        """
        Select the best agent based on the highest Sharpe ratio.
        """
        best_agent_name = max(sharpe_ratios, key=sharpe_ratios.get)
        self.current_agent = self.agents[best_agent_name]
        self.current_agent_name = best_agent_name  # Set the current agent name
        print(f"Selected {best_agent_name} as the best-performing agent.")

    def evaluate_agents(self, rewards, risk_free_rate=0.0):
        """
        Evaluate each agent based on their Sharpe ratio.
        """
        sharpe_ratios = {}
        for name, agent_rewards in rewards.items():
            avg_return = np.mean(agent_rewards)
            volatility = np.std(agent_rewards)
            sharpe_ratios[name] = (avg_return - risk_free_rate) / (volatility + 1e-6)
        return sharpe_ratios

    def act(self, state):
        """
        Use the currently selected agent to act.
        """
        if self.current_agent is None:
            raise Exception("No agent selected. Please call select_best_agent first.")
        return self.current_agent.act(state)

    def train(self, state, action, reward, next_state, done):
        """
        Train the currently selected agent with the provided experience tuple.
        """
        if self.current_agent is None:
            raise Exception("No agent selected. Please call select_best_agent first.")

        self.current_agent.train(state, action, reward, next_state, done)


    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """
        Calculate the Sharpe Ratio.

        Args:
            returns (list or np.ndarray): List or nested list of returns.
            risk_free_rate (float): Risk-free rate (default is 0.0).

        Returns:
            float: Sharpe Ratio.
        """
        # Flatten and filter non-numeric elements
        flattened_returns = []
        for r in returns:
            if isinstance(r, (list, np.ndarray)):
                flattened_returns.extend(r)
            else:
                flattened_returns.append(r)

        # Ensure all elements are floats
        try:
            returns = np.array(flattened_returns, dtype=float)
        except ValueError as e:
            print(f"Error converting returns to array: {e}")
            print(f"Returns: {flattened_returns}")
            raise

        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0  # Avoid division by zero if standard deviation is 0

        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    @staticmethod
    def simulate_agent_performance(agent, env, val_data):
        """
        Simulates the agent's performance on the validation data.
        Args:
            agent (object): The agent to simulate (e.g., DDPGAgent, PPOAgent).
            env (StockTradingEnv): The trading environment.
            val_data (np.ndarray): Validation data for the simulation.
        Returns:
            list: Simulated returns from the agent's actions.
        """
        print(f"Simulating agent performance with val_data shape: {val_data.shape}")
        env.data = val_data  # Assign reshaped validation data to the environment
        state = env.reset()  # Reset the environment to start simulation
        simulated_returns = []

        done = False
        while not done:
            # Validate state shape before reshaping
            if state.shape != (env.lookback_window_size, val_data.shape[-1]):
                raise ValueError(
                    f"State shape mismatch in simulation. Expected {(env.lookback_window_size, val_data.shape[-1])}, got {state.shape}"
                )

            # Reshape state to match agent's input requirements
            state = state.reshape(1, *state.shape)
            logger.debug(f"State shape passed to act: {state.shape}, expected state_dim: {agent.state_dim}")

            action = agent.act(state)  # Get action from the agent
            next_state, reward, done, _ = env.step(action)  # Perform action in the environment
            simulated_returns.append(reward)
            state = next_state  # Move to the next state

        return simulated_returns





