import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
import streamlit as st
import datetime
import plotly.express as px

def get_tickers():
    tickers = []
    n_tickers = st.sidebar.slider("Number of ticker:", 1, 30, 5)
    for i in range(n_tickers):
        tickers.append(st.text_input(f'Ticker {i+1}'))
    return tickers
tickers = get_tickers()

START_DATE = st.sidebar.date_input("Start date:", datetime.datetime(1980, 1, 1))
END_DATE = st.sidebar.date_input("End date:")
n_assets = len(tickers)
prices_df = yf.download(tickers, start=START_DATE, end=END_DATE, ignore_tz = True)
prices_df.index = prices_df.index.tz_localize(None)
st.title('Price History')
st.dataframe(prices_df['Adj Close'].dropna().round(3))

st.title('Returns')
returns_df = prices_df['Adj Close'].pct_change()
cumulative_returns_df = (1 + returns_df).cumprod() - 1
st.line_chart(cumulative_returns_df)
#st.line_chart(prices_df['Adj Close'])


#Matrice correlazione
st.title('Correlation Matrix')
returns_df = prices_df['Adj Close'].pct_change().dropna()
corr_mat = returns_df.corr()
plt.figure(figsize=(14, 6))
sn.heatmap(corr_mat, annot=True, cmap='RdYlGn')
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.write(pd.DataFrame(corr_mat))
st.pyplot()

N_PORTFOLIOS = 100000
N_DAYS = 252
tickers.sort()
returns_df = prices_df['Adj Close'].pct_change().dropna()
avg_returns = returns_df.mean() * N_DAYS
cov_mat = returns_df.cov()* N_DAYS 
cov_mat.round(5)
np.random.seed(88)
weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
weights /= np.sum(weights, axis=1)[:, np.newaxis]
print(weights)
portf_rtns = np.dot(weights, avg_returns)
portf_vol = []
for i in range(0, len(weights)): 
    portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))
portf_vol = np.array(portf_vol)
portf_sharpe_ratio = portf_rtns / portf_vol
portf_results_df = pd.DataFrame({'returns': portf_rtns, 'volatility': portf_vol, 'sharpe_ratio': portf_sharpe_ratio})

N_POINTS = 100
portf_vol_ef = []
indices_to_skip = []
portf_rtns_ef = np.linspace(portf_results_df.returns.min(), portf_results_df.returns.max(), N_POINTS)
portf_rtns_ef = np.round(portf_rtns_ef, 2)
portf_rtns = np.round(portf_rtns, 2)
for point_index in range(N_POINTS):
    if portf_rtns_ef[point_index] not in portf_rtns:
        indices_to_skip.append(point_index)
        continue
    matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
    portf_vol_ef.append(np.min(portf_vol[matched_ind]))
portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip) 

MARKS = ['o', 'X', 'd', '*' , 'p' ,'1','2','3','4','.','8','s','h','^','+','<','>']
fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility', y='returns', c='sharpe_ratio', cmap='RdYlGn', edgecolors='black', ax=ax)
ax.set(xlabel='Volatility', ylabel='Expected Returns', title='Efficient Frontier')
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
for asset_index in range(n_assets):
    ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]), y=avg_returns[asset_index], marker=MARKS[asset_index], 
               s=150, color='black', label=tickers[asset_index])
ax.legend()
plt.rcParams["figure.figsize"] = [16,9]
st.pyplot(fig)

def sharpe_ratio(returns, adjustment_factor=0.0):
        returns_risk_adj = returns - adjustment_factor
        return (returns_risk_adj.mean() / returns_risk_adj.std()) * np.sqrt(252)

EQUALWEIGHT, max_sharpe_ind , min_sharpe_ind, min_vol_ind, max_ret =st.tabs(['Equal Weight','Max Sharpe Indicator','Min Sharpe indicator',
                                                                             'Min volatility','Max returns'])
#EQUALWEIGHT
with EQUALWEIGHT:
    st.title('Equal weights')
    portfolio_weights = n_assets * [1 / n_assets]
    weights_dict = dict(zip(tickers, portfolio_weights))
    returns = prices_df['Adj Close'].pct_change().dropna()
    portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)
    weights_equal = [1/n_assets for i in range(n_assets)]
    figequal = px.pie(values=weights_equal, names=tickers, title='Peso delle azioni min sharpe')
    st.plotly_chart(figequal)
    pf.create_simple_tear_sheet(portfolio_returns)
    plt.plot(portfolio_returns)
    st.pyplot()
    #pf.timeseries.sharpe_ratio(portfolio_returns, risk_free=0.04/365, period='daily')


with max_sharpe_ind:
    max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
    max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
    st.title('Maximum Sharpe ratio portfolio')
    st.write('Performance')
    st.write('returns      =', round(max_sharpe_portf['returns']*100, 2),'%')
    st.write('volatility   =',round(max_sharpe_portf['volatility']*100, 2),'%')
    st.write('sharpe_ratio =',round(max_sharpe_portf['sharpe_ratio'], 3))
    st.write('\nWeights')
    weights_maxsharpe=weights[np.argmax(portf_results_df.sharpe_ratio)]
    returns = prices_df['Adj Close'].pct_change().dropna()
    portfolio_returns_maxsharpe = pd.Series(np.dot(weights_maxsharpe, returns.T), index=returns.index)
    weights_df_maxshar = pd.DataFrame({'tickers': tickers, 'weights': weights[np.argmax(portf_results_df.sharpe_ratio)]})
    figmaxshar = px.pie(weights_df_maxshar, names='tickers', values='weights', title='Peso delle azioni max sharpe')
    st.plotly_chart(figmaxshar)
    pf.create_simple_tear_sheet(portfolio_returns_maxsharpe)
    plt.plot(portfolio_returns_maxsharpe)
    st.pyplot()
    

with min_sharpe_ind:
    min_sharpe_ind = np.argmin(portf_results_df.sharpe_ratio)
    min_sharpe_portf = portf_results_df.loc[min_sharpe_ind]
    st.title('Minimum Sharpe ratio portfolio')
    st.write('Performance')
    st.write('returns      =', round(min_sharpe_portf['returns']*100, 2),'%')
    st.write('volatility   =',round(min_sharpe_portf['volatility']*100, 2),'%')
    st.write('sharpe_ratio =',round(min_sharpe_portf['sharpe_ratio'], 3))
    st.write('\nWeights')
    weights_minsharpe=weights[np.argmin(portf_results_df.sharpe_ratio)]
    returns = prices_df['Adj Close'].pct_change().dropna()
    portfolio_returns_minsharpe = pd.Series(np.dot(weights_minsharpe, returns.T), index=returns.index)
    weights_df_minshar = pd.DataFrame({'tickers': tickers, 'weights': weights[np.argmin(portf_results_df.sharpe_ratio)]})
    figminshar = px.pie(weights_df_minshar, names='tickers', values='weights', title='Peso delle azioni min sharpe')
    st.plotly_chart(figminshar)
    pf.create_simple_tear_sheet(portfolio_returns_minsharpe)
    plt.plot(portfolio_returns_minsharpe)
    st.pyplot()

with min_vol_ind:
    min_vol_ind = np.argmin(portf_results_df.volatility)
    min_vol_portf = portf_results_df.loc[min_vol_ind]
    st.title('Minimum volatility portfolio')
    st.write('Performance')
    st.write('returns      =', round(min_vol_portf['returns']*100, 2),'%')
    st.write('volatility   =',round(min_vol_portf['volatility']*100, 2),'%')
    st.write('sharpe_ratio =',round(min_vol_portf['sharpe_ratio'], 3))
    st.write('\nWeights')
    #for x, y in zip(tickers,  weights[np.argmin(portf_results_df.volatility)]):
    # #    print(f'{x}: {100*y:.2f}% ', end="", flush=True)
    weights_minvol=weights[np.argmin(portf_results_df.volatility)]
    returns = prices_df['Adj Close'].pct_change().dropna()
    portfolio_returns_minvol = pd.Series(np.dot(weights_minvol, returns.T), index=returns.index)
    weights_df_min = pd.DataFrame({'tickers': tickers, 'weights': weights[np.argmin(portf_results_df.volatility)]})
    figmin = px.pie(weights_df_min, names='tickers', values='weights', title='Peso delle azioni minimo volatilità')
    st.plotly_chart(figmin)
    pf.create_simple_tear_sheet(portfolio_returns_minvol)
    plt.plot(portfolio_returns_minvol)
    st.pyplot()

with max_ret:
    max_ret = np.argmax(portf_results_df.returns)
    max_ret_portf = portf_results_df.loc[max_ret]
    st.title('Maximum returns portfolio')
    st.write('Performance')
    st.write('returns      =', round(max_ret_portf['returns']*100, 2),'%')
    st.write('volatility   =',round(max_ret_portf['volatility']*100, 2),'%')
    st.write('sharpe_ratio =',round(max_ret_portf['sharpe_ratio'], 3))
    st.write('\nWeights')
    #for x, y in zip(tickers,  weights[np.argmax(portf_results_df.returns)]):
    #    print(f'{x}: {100*y:.2f}% ', end="", flush=True)
    weights_maxret = weights[np.argmax(portf_results_df.returns)]
    returns = prices_df['Adj Close'].pct_change().dropna()
    portfolio_max_returns= pd.Series(np.dot(weights_maxret, returns.T), index=returns.index)
    weights_df_max = pd.DataFrame({'tickers': tickers, 'weights': weights[np.argmax(portf_results_df.returns)]})
    figmax = px.pie(weights_df_max, names='tickers', values='weights', title='Peso delle azioni nel portafoglio con il massimo rendimento')
    st.plotly_chart(figmax)
    pf.create_simple_tear_sheet(portfolio_max_returns)
    plt.plot(portfolio_max_returns)
    st.pyplot()

st.empty()
st.title('Efficient frontier')

fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',y='returns', c='sharpe_ratio',cmap='RdYlGn', edgecolors='black',ax=ax)
ax.scatter(x=max_sharpe_portf.volatility,y=max_sharpe_portf.returns,c='black', marker='*',s=200, label='Max Sharpe Ratio')
ax.scatter(x=min_sharpe_portf.volatility,y=min_sharpe_portf.returns,c='black', marker='D',s=200, label='Min Sharpe Ratio')
ax.scatter(x=min_vol_portf.volatility,y=min_vol_portf.returns,c='black', marker='P',s=200, label='Minimum Volatility')
ax.scatter(x=max_ret_portf.volatility,y=max_ret_portf.returns,c='black', marker='X',s=200, label='Max return')
ax.set(xlabel='Volatility', ylabel='Expected Returns',title='Efficient Frontier')
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
ax.legend()
plt.rcParams["figure.figsize"] = [16,9]
st.pyplot(fig)


st.title('Cumulative returns')
df_port=pd.DataFrame({'1/n':portfolio_returns, 'max sharpe':portfolio_returns_maxsharpe, 'min_vol':portfolio_returns_minvol,
                     'max_return':portfolio_max_returns, 'min sharpe': portfolio_returns_minsharpe}) 
df_port_cumul = (1 + df_port).cumprod() - 1
df_port_cumul_returns = df_port_cumul.reset_index()
df2 = df_port_cumul_returns.melt(id_vars=['Date'], var_name='Portfolio', value_name='cum_return')
df2['cum_return_pct'] = df2['cum_return'] * 100
fig1 = px.line(df2, x='Date', y='cum_return_pct', color='Portfolio', title='Performance - Daily Cumulative Returns',
              labels={'cum_return_pct':'daily cumulative returns (%)', })
st.plotly_chart(fig1)
