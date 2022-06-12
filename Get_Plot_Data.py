"""
Get_Plot_Data.py
Author: Alexey Zagorulko
Date: 09.06.2022
Goal: Answer questions on how to get, organise and analyze stock data

Questions:
1. Where can I get stock data from?
2. What tools can I use to get stock data?
3. How is stock data saved?
4. What visualisations are important and how do I read them?
5. What information can I retrieve from stock data?
6. How can I store that information in a SQL database?

Other resources for quant libraries: https://github.com/wilsonfreitas/awesome-quant#data-sources
"""

import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# SET plotly as pandas plot standard
pd.options.plotting.backend = 'plotly'


""" Question 1 & 2 & 3: Where can I get stock data from, which tools can I use and how is it stored?

To import data from a source of your choice take pandas data-reader
web: https://pandas-datareader.readthedocs.io/en/latest/readers/index.html
it has access to following Data Readers:
- AlphaVantage
- Federal Reserve Economic Data (FRED)
- Fama-French Data (Ken French’s Data Library)
- Bank of Canada
- Econdb
- Enigma
- Eurostat
- The Investors Exchange (IEX)
- Moscow Exchange (MOEX)
- NASDAQ
- Naver Finance
- Organisation for Economic Co-operation and Development (OECD)
- Quandl
- Stooq.com
- Tiingo
- Thrift Savings Plan (TSP)
- World Bank
- Yahoo Finance

Before importing data, specify what you want to do with the data, the date range and goals i.e PSEUDOCODE

"""

# DATE
end = dt.datetime.now()
start = dt.datetime(2006,1,1)

# STOCKS to analyze
stocklist = ['ZURN', 'SREN', 'NESN', 'ROG']
stocks = [i + '.SW' for i in stocklist]

# GET data on stocks and display data frame structure
df = pdr.get_data_yahoo(stocks, start, end)
print(df.index)
print(df.columns)

# DEFINE sub data frames
Close = df.Close
Close100 = Close[Close.index > end - dt.timedelta(days=100)]
CloseZURN = Close['ZURN.SW'].pct_change()  # percent changes from day to day (sell-buy)/buy

# DESCRIBE function for pandas data frames gives simple statistics
print(Close.describe())
print(Close100.describe())


# PLOT stock data with plotly
#Close.plot().show()
#CloseZURN.plot(kind='hist').show()


""" Question 4: What visualisations are important, how to create and how to read them?

Candlestick graphs are important and visualise most important information on a stock:
- Open and Closing price
- High and Low price
- Volume traded 
- Performance against index
- Supply/Demand

"""
pd.options.mode.chained_assignment = None #disable false positive warning

ZURN = df.iloc[:, df.columns.get_level_values(1)=='ZURN.SW']  #GET certain columns in multi index

# MOVING AVERAGE
ZURN['MA50'] = ZURN['Close'].rolling(window=50, min_periods=0).mean()  #with set min periods of zero to avoid NaN
ZURN['MA200'] = ZURN['Close'].rolling(window=200, min_periods=0).mean()
ZURN['VA50'] = ZURN['Volume'].rolling(window=50, min_periods=0).mean()

# PLOT
# CREATE SUBPLOT
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.1,
                    subplot_titles=('ZURN','Volume'), row_width=[0.2,0.7])
# ADD TRACE
# add Candlestick
fig.add_trace(go.Candlestick(x=ZURN.index, open=ZURN[('Open','ZURN.SW')], high=ZURN[('High','ZURN.SW')],
                             low=ZURN[('Low','ZURN.SW')], close=ZURN[('Close','ZURN.SW')], name='OHLC'),row=1,col=1)
# add moving average
fig.add_trace(go.Scatter(x=ZURN.index, y=ZURN['MA50'], marker_color='grey', name='MA50'), row=1, col=1)
fig.add_trace(go.Scatter(x=ZURN.index, y=ZURN['MA200'], marker_color='lightgrey', name='MA200'), row=1, col=1)

# add volume bar chart
fig.add_trace(go.Bar(x=ZURN.index, y=ZURN[('Volume','ZURN.SW')], marker_color='red',showlegend=False), row=2,col=1)
fig.add_trace(go.Scatter(x=ZURN.index, y=ZURN['VA50'], marker_color='lightgrey', name='VA50'),row=2,col=1)

# update layout, legend, labels
fig.update_layout(
    title='ZURN Historical Price Chart',
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='Price (CHF/share)',
        title_font_size=14,
        tickfont_size=12
    ),
    autosize=False,
    width=800,
    height=500,
    margin=dict(l=50,r=50,b=100,t=100,pad=5),
    paper_bgcolor='Lightsteelblue'

)

fig.update(layout_xaxis_rangeslider_visible=False)  # blend out rangeslider on xaxis

fig.show()

"""
Question 6: How can I store the stock data in SQL?

"""