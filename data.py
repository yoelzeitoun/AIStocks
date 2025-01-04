import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates
from alpha_vantage.foreignexchange import ForeignExchange
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from Data.indicators import Indicators
import matplotlib.pyplot as plt

class Data:
    def __init__(self, asset, RSI=False, MA=False, BB=False, PP=False, FIB=False, PAT=False, drop_ohl=True, lookback=3, interval='1d'):
        self.interval = interval
        self.start = self.set_start_with_max_interval()
        self.now = datetime.now() - timedelta(days=1)
        self.asset = asset
        self.file_path = self.get_file_path(asset)

        if os.path.exists(self.file_path):
            print(f"Loading existing data for {asset}...")
            self.prices = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        else:
            print(f"Downloading data for {asset}...")
            self.prices = self.load_data(asset)
            self.prices.to_csv(self.file_path)
            print(f"Data saved to {self.file_path}")

        # self.emphasize_close_price(close_column='Close', weight_factor=10)
        self.prices = Indicators(self.prices).apply_indicators(RSI, MA, BB, PP, FIB, PAT, drop_ohl)
        self.data = DataPreprocessor(self.prices, lookback)

    def set_start_with_max_interval(self):
        now = datetime.now()
        interval_dict = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}

        if self.interval[-1] not in interval_dict:
            raise ValueError("Invalid interval format. Please use 'm', 'h', 'd', or 'w'.")

        interval_unit = interval_dict[self.interval[-1]]
        year, month, day = now.year, now.month, now.day

        if interval_unit == 'minutes':
            month -= 3
        elif interval_unit == 'hours':
            year -= 1
        elif interval_unit == 'days':
            year -= 4
        elif interval_unit == 'weeks':
            year -= 5

        return datetime(year, month, day)

    def get_file_path(self, asset):
        """Generate a file path for the asset."""
        base_path = r"C:\Users\yoelarie\OneDrive - Intel Corporation\Desktop\תואר שני\Project AI"
        file_name = f"{asset}_data.csv"
        return os.path.join(base_path, file_name)

    def get_intc_data(self):
        INTC = yf.download("INTC", start=self.start, end=self.now, interval=self.interval)
        INTC.drop(columns=['Adj Close', 'Volume'], inplace=True)
        return INTC

    def get_eurusd_data(self):
        api_key = 'OVFGKLE6XTY10LFS'
        fx = ForeignExchange(key=api_key)

        data, _ = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')
        EURUSD = pd.DataFrame.from_dict(data, orient='index')
        EURUSD.index = pd.to_datetime(EURUSD.index)
        EURUSD.sort_index(inplace=True)

        EURUSD.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        }, inplace=True)

        EURUSD.drop(columns=['5. volume'], inplace=True, errors='ignore')

        start_date = '2010-01-01'
        end_date = '2024-06-14'
        EURUSD = EURUSD.loc[start_date:end_date]
        return EURUSD

    def get_gold_data(self):
        GOLD = yf.download("GC=F", start=self.start, end=self.now, interval=self.interval)
        GOLD.drop(columns=['Adj Close', 'Volume'], inplace=True)
        return GOLD

    def get_btc_data(self):
        BTC = yf.Ticker("BTC-USD").history(start=self.start)
        BTC = BTC.drop(columns=['Volume', 'Dividends', 'Stock Splits'])
        return BTC

    def load_data(self, asset):
        asset_methods = {
            'INTC': self.get_intc_data,
            'EURUSD': self.get_eurusd_data,
            'GOLD': self.get_gold_data,
            'BTC': self.get_btc_data
        }
        if asset not in asset_methods:
            raise ValueError(f"Invalid asset name '{asset}'.")
        return asset_methods[asset]()

    def emphasize_close_price(self, close_column='Close', weight_factor=10):
        self.prices[close_column + '_weighted'] = self.prices[close_column] * weight_factor

    def plot_full_data(self):
        """Plots the entire dataset with available indicators."""
        plt.figure(figsize=(14, 7))  # Set figure size
        for column in self.prices.columns:
            plt.plot(self.prices.index, self.prices[column], label=column)
        plt.title('Full Data Chart')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.show()



# Data Preprocessing and Scaling
class DataPreprocessor:
    def __init__(self, prices, lookback):
        self.prices = prices
        if np.any(self.prices == 0):
            print("[ERROR] Zero price found in initial data! This should be corrected.")

        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0.0001, 1))

        self.train_prices, self.test_prices = self.split_data()
        self.train_prices_scaled, self.test_prices_scaled = self.scale_data(self.train_prices, self.test_prices)
        self.x_train, self.y_train, self.x_test, self.y_test = self.create_all_sequences(
            self.train_prices_scaled, self.test_prices_scaled
        )
        self.print_shapes()

    def split_data(self):
        test_size = int(len(self.prices) / 6)
        train_size = len(self.prices) - test_size
        print("train_size: ", train_size)
        print("test_size: ", test_size)

        train_prices = self.prices.iloc[:train_size, :]
        test_prices = self.prices.iloc[train_size:, :]

        return train_prices, test_prices

    def scale_data(self, train_prices, test_prices):
        train_scaled = self.scaler.fit_transform(train_prices)
        test_scaled = self.scaler.transform(test_prices)

        train_prices_scaled = pd.DataFrame(train_scaled, index=train_prices.index, columns=train_prices.columns)
        test_prices_scaled = pd.DataFrame(test_scaled, index=test_prices.index, columns=test_prices.columns)

        return train_prices_scaled, test_prices_scaled

    def create_sequences(self, data):
        x, y = [], []
        for i in range(len(data) - self.lookback):
            x.append(data.iloc[i:i + self.lookback].values)
            y.append(data.iloc[i + self.lookback]['Close'])
        return np.array(x), np.array(y)

    def create_all_sequences(self, train_prices, test_prices):
        x_train, y_train = self.create_sequences(train_prices)
        x_test, y_test = self.create_sequences(test_prices)
        return x_train, y_train, x_test, y_test

    def print_shapes(self):
        print("x_train.shape", self.x_train.shape)
        print("y_train.shape", self.y_train.shape)
        print("x_test.shape", self.x_test.shape)
        print("y_test.shape", self.y_test.shape)
