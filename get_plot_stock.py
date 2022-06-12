"""
get_plot_stock.py
Author: Alexey Zagorulko
Date: 09.06.2022

Function to GET and PLOT stock data from Yahoo
Input: Stock name on yahoo finance, start date, end date
Output: dataframe with stock data and candle stick graph including:
- OHLC
- MA50 (moving average 50)
- MA200 (moving average 200)
- VA (volume moving average 50)
- Volume

"""

import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SET plotly as pandas plot standard
pd.options.plotting.backend = 'plotly'


def get_plot_stock(stock, start, end):
    df = pdr.get_data_yahoo(stock, start, end)
    # MOVING AVERAGE TERMS
    df['MA50'] = df['Close'].rolling(window=50, min_periods=0).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=0).mean()
    df['VA50'] = df['Volume'].rolling(window=50, min_periods=0).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(stock, 'Volume'), row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    # add moving average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], marker_color='grey', name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], marker_color='lightgrey', name='MA200'), row=1, col=1)

    # add volume bar chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='red', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VA50'], marker_color='lightgrey', name='VA50'), row=2, col=1)

    # update layout, legend, labels
    fig.update_layout(
        title=stock + ' Historical Price Chart',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Price (Currency/share)',
            title_font_size=14,
            tickfont_size=12
        ),
        autosize=False,
        width=800,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=5),
        paper_bgcolor='Lightsteelblue'

    )

    fig.update(layout_xaxis_rangeslider_visible=False)  # blend out rangeslider on xaxis

    return df, fig
