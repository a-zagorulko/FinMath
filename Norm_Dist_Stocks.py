"""
Norm_Dist_Stocks.py
Author: a-zagorulko
Date: 08.06.2022

Goal: Answer questions regarding how to statistically view stock data and returns

Questions:
1. How can I calculate stock returns and what do they tell me?
2. How can I model stock returns?
3. Are stock returns normally distributed and how can I test that?

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
Question 1: How can I calculate stock returns? 

"""

stock = 'ZURN.SW'
end = dt.datetime.now()
start = dt.datetime(2006, 1, 1)

df, fig = get_plot_stock(stock, start, end)

"""
Question 2: How can I model stock returns?

Idea 1: We assume stock returns are normally distributed, but we need to use log returns.

Simple Returns: the product of normally distributed variables is not normal

R_{t} = {P_{t}-P_{t-1}}/P_{t-1} = P{t}/P_{t-1}-1
1 + R_{t} = P_{t}/P_{t-1}

k-period simple returns:
1 + R_{t}(k) = \prod_{i=0}^{k-1} {(1+R_{i-1})}

Log Returns: the sum of normally distributed variables is normal

k-period log returns: 

r_{t}(k) = ln(1+r_{t}(k)) = r_{t}+...+r_{t+k-1} = ln(P_{t})-ln(P_{t-k})

"""

simple_returns = df.Close.pct_change().dropna()
# LIMITS of simple returns mean shown. Beginning price with returns mean compounded is NOT a descriptive statistic.
print(df.Close[0] * (1+simple_returns.mean())**len(simple_returns)==df.Close[-1])
print(df.Close[0] * np.prod([(1+Rt) for Rt in simple_returns]))
print(df.Close[-1])

# ADDITIVE PROPERTY OF LOG RETURNS
log_returns = np.log(df.Close/df.Close.shift(1)).dropna()
log_returns_mean = log_returns.mean()
print(df.Close[0] * np.exp(log_returns_mean*len(log_returns)))

# HISTOGRAM
log_returns.plot(kind='hist').update_layout(autosize=False, width=800, height=500).show()

""" 
Question 3: Are log returns normally distributed and is this a good assumption?
"""
log_returns_sorted = log_returns.tolist()
log_returns_sorted.sort()
worst = log_returns_sorted[0]
best = log_returns_sorted[-1]

# NORMALISE
std_worst = (worst - log_returns.mean()/log_returns.std())
std_best = (best - log_returns.mean()/log_returns.std())

print('Std dev. worst %.2f best %.2f' %(std_worst, std_best))
# Probability of having this standard deviation
print('Prob worst %.10f best %.10f' %(stats.norm(0,1).pdf(std_worst), stats.norm(0,1).pdf(best)))

# TEST ASSUMPTION OF NORMALITY
# https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411

# Q-Q Plot against normal distribution
stats.probplot(log_returns, dist='norm', plot=pylab)
plt.show()  # we get long tails

# BOX Plot
log_returns.plot(kind='box').show()

# KOLMOGOROV-SMIRNOV TEST computes the distances between empirical distribution and theoretical distribution
# If KS Test value = 0 then the distributions are the same
# Null Hypothesis: samples are from the same distribution
ks_stat, p_value = stats.kstest(log_returns, 'norm')
print(ks_stat, p_value)
if p_value > 0.05:
    print('Probability is Gaussian')
else:
    print('Probability is NOT Gaussian')

# SHAPIRO-WILK TEST for normal distributions
sw_stat, p_value = stats.shapiro(log_returns)
print(sw_stat, p_value)
if p_value > 0.05:
    print('Probability is Gaussian')
else:
    print('Probability is NOT Gaussian')

"""
Financial data in short windows can be assumed to be normally distributed, but in big windows it is not.
FInding a time window where it is acceptably normally distributed allows for using techniques in max window.
"""


