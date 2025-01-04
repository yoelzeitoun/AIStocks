# import numpy as np
#
#
# def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
#     """
#     Calculate the Sharpe ratio for a given set of returns.
#
#     Parameters:
#     - returns (numpy.array): Array of returns
#     - risk_free_rate (float): Risk-free rate, default is 0.0
#
#     Returns:
#     - float: Sharpe Ratio
#     """
#     # Calculate the difference in returns and the risk-free rate
#     excess_returns = returns - risk_free_rate
#
#     # Mean of excess returns
#     mean_excess_returns = np.mean(excess_returns)
#
#     # Standard deviation of returns
#     std_dev_returns = np.std(returns)
#
#     # Sharpe ratio calculation
#     sharpe_ratio = mean_excess_returns / std_dev_returns if std_dev_returns != 0 else 0
#
#     return sharpe_ratio
