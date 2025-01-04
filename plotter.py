import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json

class PredictionVisualizer:
    def __init__(self, model, data, rewards=None, portfolio_values=None, reward_timestamps=None, initial_portfolio=10000, prices=None, plot=True):
        self.model = model
        self.data = data
        self.scaler = data.scaler
        self.num_features = data.x_train.shape[2]
        self.lookback = data.lookback
        self.rewards = rewards  # Store rewards for visualization
        self.portfolio_values = portfolio_values  # New parameter
        self.reward_timestamps = reward_timestamps  # Store timestamps for reward visualization
        self.initial_portfolio = initial_portfolio  # Starting portfolio value
        self.prices = prices  # Pass the prices for calculation

        # Plot portfolio evolution for train and test datasets
        if plot and self.portfolio_values is not None:
            self.plot_all_portfolios()

        self.save_run_details()

    def generate_predictions(self, x, target_index):
        y_pred = self.model.predict(x)

        # Flatten predictions if 2D
        if y_pred.ndim == 2:
            y_pred = y_pred.flatten()

        # Reduce 3D predictions to 2D by taking the last time step
        if y_pred.ndim == 3:
            y_pred = y_pred[:, -1, :]

        y_pred = y_pred.reshape(-1, 1)

        # Pad to match scaler input
        num_features = self.scaler.n_features_in_
        y_pred_scaled = np.hstack([np.zeros((len(y_pred), num_features - 1)), y_pred])

        # Inverse transform to get original scale
        y_pred_original = self.scaler.inverse_transform(y_pred_scaled)[:, -1]
        return y_pred_original[:len(target_index)]

    def generate_all_predictions(self):
        y_train_pred_original = self.generate_predictions(self.data.x_train, self.train_prices.index[self.lookback:])
        y_test_pred_original = self.generate_predictions(self.data.x_test, self.test_prices.index[self.lookback:])
        return y_train_pred_original, y_test_pred_original

    def get_original_prices(self):
        y_train_original = self.data.scaler.inverse_transform(
            np.hstack([self.data.train_prices_scaled.iloc[self.data.lookback:, :-1],
                       self.data.y_train.reshape(-1, 1)]))[:, -1]
        y_test_original = self.data.scaler.inverse_transform(
            np.hstack([self.data.test_prices_scaled.iloc[self.data.lookback:, :-1],
                       self.data.y_test.reshape(-1, 1)]))[:, -1]
        return y_train_original, y_test_original

    def save_run_details(self, filename="model_run_details.json"):
        """
        Save run details to a JSON file, including total portfolio gain for the Test phase.
        """
        # Compute total gain from Test portfolio values
        test_portfolio = self.portfolio_values.get("Test", [])
        if len(test_portfolio) > 0:
            total_gain = test_portfolio[-1] - self.initial_portfolio  # Gain = Final - Initial portfolio value
        else:
            total_gain = 0  # Default if no portfolio values exist

        run_details = {
            "model_type": getattr(self.model, 'name', 'Unknown Model'),
            "input_shape": self.model.input_shape,
            "total_gain": total_gain  # Add total gain for Test data
        }

        try:
            with open(filename, "r+") as file:
                data = json.load(file)
                data.append(run_details)
                file.seek(0)
                json.dump(data, file, indent=4)
        except FileNotFoundError:
            with open(filename, "w") as file:
                json.dump([run_details], file, indent=4)

        print(f"Run details saved: {run_details}")

    def calculate_portfolio(self):
        """
        Calculate real portfolio values including cash balance and shares held at each step.
        """
        if not self.rewards or len(self.rewards) == 0 or self.prices is None:
            print("Warning: Rewards or prices are missing.")
            return [self.initial_portfolio], []

        portfolio_values = [self.initial_portfolio]
        transaction_results = []

        current_balance = self.initial_portfolio  # Initial cash
        shares_held = 0  # Initially no shares held

        print(f"Initial Portfolio Value: {self.initial_portfolio:.2f}")

        for i, reward in enumerate(self.rewards):
            # Ensure price index does not exceed available prices
            if i < len(self.prices):
                current_price = self.prices[i]
            else:
                current_price = self.prices[-1]  # Fallback to the last known price
                print(f"Warning: Missing price for step {i}, using last known price: {current_price:.2f}")

            # Update balance and portfolio
            current_balance += reward
            portfolio_value = current_balance + (shares_held * current_price)
            portfolio_values.append(portfolio_value)

            # Debugging output
            print(f"Step {i + 1}: Reward = {reward:.2f}, Shares Held = {shares_held}, "
                  f"Current Price = {current_price:.2f}, Portfolio Value = {portfolio_value:.2f}")

        return portfolio_values[1:], transaction_results  # Exclude initial value

    def plot_all_portfolios(self):
        """
        Plot portfolio values for Train and Test datasets in one window.
        """
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
        datasets = [
            (self.prices["Train"], self.portfolio_values["Train"], "Train", axes[0]),
            (self.prices["Test"], self.portfolio_values["Test"], "Test", axes[1]),
        ]

        for prices, portfolio_values, dataset_name, ax in datasets:
            dates = range(len(prices))  # Generate indices as x-axis
            min_length = min(len(dates), len(portfolio_values))
            dates = dates[:min_length]
            portfolio_values = portfolio_values[:min_length]

            # Plot portfolio values
            ax.plot(dates, portfolio_values, label=f"{dataset_name} Portfolio Value", color="green")

            ax.set_title(f"{dataset_name} Portfolio Value Over Time")
            ax.set_ylabel("Portfolio Value")
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Steps")
        plt.tight_layout()
        plt.show()
