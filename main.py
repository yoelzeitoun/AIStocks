from data import Data
from models import ModelBuilder, Model

# Choose between 'INTC' 'EURUSD' 'GOLD' 'BTC'
# data = Data('GOLD', RSI=True, MA=True, BB=True, PP=True, FIB=True, drop_ohl=False, lookback=3).data
data = Data('GOLD', RSI=False, MA=False, BB=False, PP=False, FIB=False, PAT=False, drop_ohl=True, lookback=5).data
# ModelBuilder(data, model_type=Model.MLP, plot=True)

# ModelBuilder(data, model_type=Model.TCN, plot=True)

# ModelBuilder(data, model_type=Model.CONV2D)

#ModelBuilder(data, model_type=Model.GRU, plot=True)

ModelBuilder(data, model_type=Model.LSTM, plot=True)