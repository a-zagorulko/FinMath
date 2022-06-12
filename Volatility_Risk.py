"""
Volatility_Risk.py
Author: a-zagorulko
Date: 11.06.2022

Goal: Answer questions regarding how volatility of stocks is measured

Questions:
1. How is volatility of one stock measured?
2. How is volatility of many stocks together measured?
3. How can I balance risk and return and what ratios are there?

"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn as sns  # statistical graphics in python
import scipy.stats as stats
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from get_plot_stock import get_plot_stock

"""
Question 1: How is volatility of a stock measured?

Taking the distribution of stock data, we measure the volatility of a stock using the standard deviation. 
Meaning we say the risk of the stock is proportional to how the stock returns deviate from the mean.

Step 1: Get Data
Step 2: Calculate Log returns
Step 3: Calculate daily_std
Step 4: Calculate annualized std
Step 5: Calculate annualized volatility
Step 6: Plot Histogram of log return frequency with annualized volatility

"""
end = dt.datetime.now()
start = dt.datetime(2014, 1, 1)

# STOCKS to analyze
stocklist = ['ZURN', 'SREN', 'NESN', 'ROG']
stocks = ['^GSPC'] + [i + '.SW' for i in stocklist]

df = pdr.get_data_yahoo(stocks, start, end)
Close = df.Close

# Compute log returns
log_returns = np.log(Close / Close.shift(1)).dropna()

# Calculate std
daily_std = log_returns.std()
annualized_vol = daily_std * np.sqrt(252) * 100

# Plot Histogram of annualized volatility
fig = make_subplots(rows=2, cols=2)

trace0 = go.Histogram(x=log_returns['ZURN.SW'], name='ZURN.SW')
trace1 = go.Histogram(x=log_returns['SREN.SW'], name='SREN.SW')
trace2 = go.Histogram(x=log_returns['NESN.SW'], name='NESN.SW')
trace3 = go.Histogram(x=log_returns['ROG.SW'], name='ROG.SW')

fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 1, 2)
fig.add_trace(trace2, 2, 1)
fig.add_trace(trace3, 2, 2)

fig.update_layout(title='Frequency of log returns',
                  xaxis=dict(title='ZURN.SW Annualized volatility:' + str(np.round(annualized_vol['ZURN.SW'], 1))),
                  xaxis2=dict(title='SREN.SW Annualized volatility:' + str(np.round(annualized_vol['SREN.SW'], 1))),
                  xaxis3=dict(title='NESN.SW Annualized volatility:' + str(np.round(annualized_vol['NESN.SW'], 1))),
                  xaxis4=dict(title='ROG.SW Annualized volatility:' + str(np.round(annualized_vol['ROG.SW'], 1)))
                  )
fig.show()

# TRAILING VOLATILITY
trading_days = 60
volatility = log_returns.rolling(window=trading_days).std() * np.sqrt(trading_days)
volatility.plot().show()

"""
Question 3: How can I get a relationship between volatility and returns?

- Sharpe Ratio (calculating risk-adjusted return with a risk free rate)
- Sortino Ratio
- Modigliani Ratio M2

"""

# SHARPE RATIO
Rf = 0.01 / 252  # risk free rate over a year
sharpe_ratio = (log_returns.rolling(
    window=trading_days).mean() - Rf) * trading_days / volatility  # volatility is calculated over trading_days
sharpe_ratio.plot().show()

# SORTINO RATIO
# not all volatility is bad, only downside volatility is bad. Upside volatility is good.
# only take downside volatility into consideration
sortino_vol = log_returns[log_returns < 0].rolling(window=trading_days, center=True, min_periods=10).std() * np.sqrt(
    trading_days)
sortino_ratio = (log_returns.rolling(window=trading_days).mean() - Rf) * trading_days / sortino_vol
sortino_vol.plot().show()
sortino_ratio.plot().show()

# MODIGLIANI RATIO M2
# measures the returns of a portfolio, adjusted for the risk of the portfolio relative to some benchmark
m2_ratio = pd.DataFrame()
benchmark_vol = volatility['^GSPC']
for c in log_returns.columns:
    if c != '^GSPC':
        m2_ratio[c] = (sharpe_ratio[c] * benchmark_vol / trading_days + Rf) * trading_days
m2_ratio.plot().show()


# MAX DRAWDOWN quantifies the steepest decline from peak to trough observed for an investment
# it does not rely on the assumption returns being of normally distributed
def max_drawdown(returns):
    cummulative_returns = (1 + returns).cumprod()  # Return cumulative product over a DataFrame or Series axis.
    peak = cummulative_returns.expanding(
        min_periods=1).max()  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.expanding.html?highlight=expanding#pandas.DataFrame.expanding
    drawdown = cummulative_returns / peak - 1
    return drawdown.min()


returns = df.Close.pct_change().dropna()
max_drawdowns = returns.apply(max_drawdown, axis=0)  # apply function to dataframe
print(max_drawdowns*100)

# CALMAR RATIO uses max drawdowns in the denominator opposed to standard deviation
calmars = np.exp(log_returns.mean()*252)/max_drawdowns  # absolute return over 252 trading days/maxdrawdowns
calmars.plot(kind='bar').show()
