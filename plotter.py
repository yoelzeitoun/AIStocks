import numpy as np
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json

# Prediction and Visualization
class PredictionVisualizer:
    def __init__(self, model, data, plot):
        self.model = model
        self.data = data
        self.scaler = data.scaler
        self.num_features = data.x_train.shape[2]
        self.lookback = data.lookback
        self.train_prices = data.train_prices
        self.val_prices = data.val_prices
        self.test_prices = data.test_prices
        self.y_train_pred_original, self.y_val_pred_original, self.y_test_pred_original = self.generate_all_predictions()
        self.y_train_original, self.y_val_original, self.y_test_original = self.get_original_prices()

        assert self.train_prices.index[self.lookback:].shape[0] == self.y_train_pred_original.shape[0]
        if plot:
            self.plot_predictions(self.y_train_original, self.y_val_original, self.y_test_original, self.y_train_pred_original,
                                  self.y_val_pred_original, self.y_test_pred_original)
        self.metrics = self.compute_metrics(self.y_train_original, self.y_val_original, self.y_test_original, self.y_train_pred_original,
                                            self.y_val_pred_original, self.y_test_pred_original)
        self.print_metrics(self.metrics)
        self.save_run_details(self.metrics)

    def generate_predictions(self, x):
        # Generate model predictions
        y_pred = self.model.model.predict(x)

        # Flatten predictions if 2D
        if y_pred.ndim == 2:
            y_pred = y_pred.flatten()  # Flatten predictions to 1D

        # Reduce 3D predictions to 2D by taking the last time step
        if y_pred.ndim == 3:
            y_pred = y_pred[:, -1, :]  # Take the last time step from 3D array

        # Ensure y_pred is 2D for further processing
        y_pred = y_pred.reshape(-1, 1)

        # Create scaled array by zero-padding
        y_pred_scaled = np.hstack([np.zeros((len(y_pred), self.num_features - 1)), y_pred])

        # Inverse transform to get original scale
        y_pred_original = self.scaler.inverse_transform(y_pred_scaled)[:, -1]

        return y_pred_original

    def generate_all_predictions(self):
        y_train_pred_original = self.generate_predictions(self.data.x_train)#.reshape(-1, self.num_features)
        y_val_pred_original = self.generate_predictions(self.data.x_val)#.reshape(-1, self.num_features)
        y_test_pred_original = self.generate_predictions(self.data.x_test)#.reshape(-1, self.num_features)
        return y_train_pred_original, y_val_pred_original, y_test_pred_original

    def get_original_prices(self):
        y_train_original = self.data.scaler.inverse_transform(
            np.hstack([self.data.train_prices_scaled.iloc[self.data.lookback:, :-1], self.data.y_train.reshape(-1, 1)]))[:, -1]
        y_val_original = self.data.scaler.inverse_transform(
            np.hstack([self.data.val_prices_scaled.iloc[self.data.lookback:, :-1], self.data.y_val.reshape(-1, 1)]))[:, -1]
        y_test_original = self.data.scaler.inverse_transform(
            np.hstack([self.data.test_prices_scaled.iloc[self.data.lookback:, :-1], self.data.y_test.reshape(-1, 1)]))[:, -1]
        return y_train_original, y_val_original, y_test_original

    def plot_predictions(self, y_train_original, y_val_original, y_test_original, y_train_pred_original,
                         y_val_pred_original, y_test_pred_original):
        plt.figure(figsize=(15, 9))

        plt.subplot(3, 1, 1)
        plt.plot(self.train_prices.index[self.lookback:], y_train_original, label='Actual Train Prices', color='blue')
        plt.plot(self.train_prices.index[self.lookback:], y_train_pred_original, label='Predicted Train Prices',
                 color='red')
        plt.title('Training Data: Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.val_prices.index[self.lookback:], y_val_original, label='Actual Validation Prices', color='blue')
        plt.plot(self.val_prices.index[self.lookback:], y_val_pred_original, label='Predicted Validation Prices',
                 color='red')
        plt.title('Validation Data: Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.test_prices.index[self.lookback:], y_test_original, label='Actual Test Prices', color='blue')
        plt.plot(self.test_prices.index[self.lookback:], y_test_pred_original, label='Predicted Test Prices',
                 color='red')
        plt.title('Test Data: Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def compute_metrics(self, y_train_original, y_val_original, y_test_original, y_train_pred_original,
                        y_val_pred_original, y_test_pred_original):
        mae_train = mean_absolute_error(y_train_original, y_train_pred_original)
        mae_val = mean_absolute_error(y_val_original, y_val_pred_original)
        mae_test = mean_absolute_error(y_test_original, y_test_pred_original)

        mse_train = mean_squared_error(y_train_original, y_train_pred_original)
        mse_val = mean_squared_error(y_val_original, y_val_pred_original)
        mse_test = mean_squared_error(y_test_original, y_test_pred_original)

        rmse_train = np.sqrt(mse_train)
        rmse_val = np.sqrt(mse_val)
        rmse_test = np.sqrt(mse_test)

        return mae_train, mse_train, rmse_train, mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test

    def print_metrics(self, metrics):
        print(f"Metrics for {self.model.model_type}:")
        print(f"Training Data - RMSE: {metrics[2]:.4f}")
        print(f"Validation Data - RMSE: {metrics[5]:.4f}")
        print(f"Test Data - RMSE: {metrics[8]:.4f}")

    def save_run_details(self, metrics, filename="model_run_details.json"):
        run_details = {
            "model_type": self.model.model_type.name,
            "input_shape": self.model.input_shape,
            # "epochs": self.epochs,
            # "batch_size": self.batch_size,
            # "learning_rate": self.learning_rate,
            # "optimizer": self.optimizer,
            "layers": [layer.get_config() for layer in self.model.model.layers],
            "train_rmse": metrics[2],
            "val_rmse": metrics[5],
            "test_rmse": metrics[8]
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

