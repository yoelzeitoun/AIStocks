import pandas as pd
import pandas_ta as ta
import talib

class Indicators:
    def __init__(self, data):
        self.data = data

    def calculate_rsi(self):
        self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)#ta.rsi(self.data['Close']) / 100

    def calculate_ma(self):
        ma_days = [20, 50]
        for ma in ma_days:
            column_name = f"MA_{ma}"
            self.data[column_name] = self.data['Close'].rolling(ma).mean()

    def calculate_bb(self):
        self.data['MA'] = self.data['Close'].rolling(window=20).mean()
        self.data['STD_20'] = self.data['Close'].rolling(window=20).std()

        self.data['Upper_BB'] = self.data['MA'] + 2 * self.data['STD_20']
        self.data['Lower_BB'] = self.data['MA'] - 2 * self.data['STD_20']
        self.data.drop('MA', axis=1, inplace=True)

    def calculate_pp(self):
        self.data['Pivot_Point'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['Support1'] = (2 * self.data['Pivot_Point']) - self.data['High']
        self.data['Resistance1'] = (2 * self.data['Pivot_Point']) - self.data['Low']
        self.data['Support2'] = self.data['Pivot_Point'] - (self.data['High'] - self.data['Low'])
        self.data['Resistance2'] = self.data['Pivot_Point'] + (self.data['High'] - self.data['Low'])
        self.data['Support3'] = self.data['Low'] - 2 * (self.data['High'] - self.data['Pivot_Point'])
        self.data['Resistance3'] = self.data['High'] + 2 * (self.data['Pivot_Point'] - self.data['Low'])

    def calculate_fib_levels(self, lookback_period):
        fib_levels = pd.DataFrame(index=self.data.index)
        fib_levels['Fib_23.6'] = self.data['High'].rolling(window=lookback_period, min_periods=lookback_period).apply(lambda x: x.max() - (x.max() - x.min()) * 0.236)
        fib_levels['Fib_38.2'] = self.data['High'].rolling(window=lookback_period, min_periods=lookback_period).apply(lambda x: x.max() - (x.max() - x.min()) * 0.382)
        fib_levels['Fib_50.0'] = self.data['High'].rolling(window=lookback_period, min_periods=lookback_period).apply(lambda x: x.max() - (x.max() - x.min()) * 0.5)
        fib_levels['Fib_61.8'] = self.data['High'].rolling(window=lookback_period, min_periods=lookback_period).apply(lambda x: x.max() - (x.max() - x.min()) * 0.618)
        fib_levels['Fib_100'] = self.data['Low'].rolling(window=lookback_period, min_periods=lookback_period).min()
        return fib_levels

    def calculate_fib(self):
        lookback_period = 100
        fib_levels = self.calculate_fib_levels(lookback_period)

        fib_columns = ['Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Fib_100']
        self.data.drop(columns=fib_columns, errors='ignore', inplace=True)

        self.data = self.data.join(fib_levels)

    def add_patterns(self):
        data = self.data['Open'], self.data['High'], self.data['Low'], self.data['Close']

        self.data['Bullish_Engulfing'] = talib.CDLENGULFING(*data)
        self.data['Bearish_Engulfing'] = talib.CDLENGULFING(*data) * -1  # Example of Bearish pattern

        # Add other patterns
        self.data['Doji'] = talib.CDLDOJI(*data)
        self.data['Hammer'] = talib.CDLHAMMER(*data)
        self.data['Shooting_Star'] = talib.CDLSHOOTINGSTAR(*data)
        self.data['Morning_Star'] = talib.CDLMORNINGSTAR(*data)
        self.data['Evening_Star'] = talib.CDLEVENINGSTAR(*data)
        self.data['Hanging_Man'] = talib.CDLHANGINGMAN(*data)
        self.data['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(*data)
        self.data['Three_Black_Crows'] = talib.CDL3BLACKCROWS(*data)
        self.data['Three_White_Soldiers'] = talib.CDL3WHITESOLDIERS(*data)
        self.data['Dark_Cloud_Cover'] = talib.CDLDARKCLOUDCOVER(*data)
        self.data['Piercing_Line'] = talib.CDLPIERCING(*data)
        self.data['Spinning_Top'] = talib.CDLSPINNINGTOP(*data)
        self.data['Marubozu'] = talib.CDLMARUBOZU(*data)
        self.data['Harami'] = talib.CDLHARAMI(*data)
        self.data['Harami_Cross'] = talib.CDLHARAMICROSS(*data)
        self.data['Three_Inside_Up'] = talib.CDL3INSIDE(*data)
        self.data['Three_Outside_Up'] = talib.CDL3OUTSIDE(*data)
        self.data['Three_Stars_In_The_South'] = talib.CDL3STARSINSOUTH(*data)
        self.data['Belt_Hold'] = talib.CDLBELTHOLD(*data)
        self.data['Counterattack'] = talib.CDLCOUNTERATTACK(*data)
        self.data['Dragonfly_Doji'] = talib.CDLDRAGONFLYDOJI(*data)
        self.data['Gravestone_Doji'] = talib.CDLGRAVESTONEDOJI(*data)
        self.data['Long_Legged_Doji'] = talib.CDLLONGLEGGEDDOJI(*data)
        self.data['Rickshaw_Man'] = talib.CDLRICKSHAWMAN(*data)
        self.data['Stalled_Pattern'] = talib.CDLSTALLEDPATTERN(*data)
        self.data['Tasuki_Gap'] = talib.CDLTASUKIGAP(*data)
        self.data['Unique_3_River'] = talib.CDLUNIQUE3RIVER(*data)
        self.data['Upside_Gap_2_Crows'] = talib.CDLUPSIDEGAP2CROWS(*data)
        self.data['Xside_Gap_3_Methods'] = talib.CDLXSIDEGAP3METHODS(*data)

    def apply_indicators(self, RSI=False, MA=False, BB=False, PP=False, FIB=False, PAT= False, drop_ohl=True):
        if RSI:
            self.calculate_rsi()
        if MA:
            self.calculate_ma()
        if BB:
            self.calculate_bb()
        if PP:
            self.calculate_pp()
        if FIB:
            self.calculate_fib()
        if PAT:
            self.add_patterns()

        self.data.dropna(inplace=True)

        if drop_ohl:
            self.data.drop(columns=['Open', 'High', 'Low'], inplace=True)

        return self.data
