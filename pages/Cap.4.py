import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
import streamlit as st
import datetime
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.simplefilter(action='ignore', category=InterpolationWarning)

RISKY_ASSET=st.text_input('Ticker')
START_DATE = st.sidebar.date_input("Start date:", datetime.datetime(1980, 1, 1))
END_DATE = st.sidebar.date_input("End date:")
df = yf.download(RISKY_ASSET, start=START_DATE,end=END_DATE)
df.index = df.index.tz_localize(None)
adj_close = df['Adj Close']
returns = adj_close.pct_change().dropna()
st.text(f'Average return: {365*100 * returns.mean():.2f}%')
fig = px.line(returns, x=returns.index, y=returns.values, labels={'x': 'Data', 'y': 'Ritorni'})
fig.update_layout(title=f'{RISKY_ASSET} returns: {START_DATE} - {END_DATE}')
st.plotly_chart(fig)

train = returns['2000-01-01':'2021-12-31']
test = returns['2022-01-01':'2022-12-31']
st.title('Test')
st.dataframe(test)

T = len(test)
N = len(test)
S_0 = adj_close[train.index[-1]] 
N_SIM = 1000  
mu = train.mean()
sigma = train.std()

def simulate_gbm(s_0, mu, sigma, n_sims, T, N, antithetic_var=False):
    dt = T/N
    if antithetic_var:
        dW_ant = np.random.normal(scale = np.sqrt(dt), size=(int(n_sims/2), N + 1))
        dW = np.concatenate((dW_ant, -dW_ant), axis=0)
    else:
        dW = np.random.normal(scale = np.sqrt(dt), size=(n_sims, N + 1))
    
    S_t = s_0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * dW, axis=1))
    S_t[:, 0] = s_0
    return S_t
gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)
LAST_TRAIN_DATE = train.index[-1].date()
FIRST_TEST_DATE = test.index[0].date()
LAST_TEST_DATE = test.index[-1].date()
PLOT_TITLE = (f'{RISKY_ASSET} Simulation 'f'({FIRST_TEST_DATE}:{LAST_TEST_DATE})')
selected_indices = adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE].index
index = [date.date() for date in selected_indices]
gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations), index=index)

ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
line_1, = ax.plot(index, gbm_simulations_df.mean(axis=1), color='red')
line_2, = ax.plot(index, adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE], color='blue')
line_3, = ax.plot(index, gbm_simulations_df.median(axis=1), color='black')
ax.set_title(PLOT_TITLE, fontsize=16)
ax.legend((line_1, line_2, line_3), ('mean', 'actual', 'median'))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


def get_tickers_and_shares():
    tickers = []
    shares = []
    n_tickers = st.sidebar.slider("Number of tickers:", 1, 30, 5)
    for i in range(n_tickers):
        ticker = st.text_input(f'Ticker {i+1}')
        share = st.number_input(f'Number of shares for {ticker}', min_value=1, max_value=100, value=1, step=1)
        tickers.append(ticker)
        shares.append(share)
    return tickers, shares, n_tickers
tickers, shares, n_tickers = get_tickers_and_shares()
    
#START_DATE = st.sidebar.date_input("Start date:", datetime.datetime(1980, 1, 1))
#END_DATE = st.sidebar.date_input("End date:")
t = 1
N_SIMS = 10000000
df = yf.download(tickers, start=START_DATE, end=END_DATE, ignore_tz = True)
adj_close = df['Adj Close'].dropna()
returns = adj_close.pct_change().dropna()
st.title('Returns')
st.dataframe(returns)

st.title('Correlation matrix')
corr_mat = returns.corr()
st.dataframe(corr_mat)

st.title('Cholesky matrix')
chol_mat = np.linalg.cholesky(corr_mat)
st.dataframe(chol_mat)

rv = np.random.normal(size=(N_SIMS, len(tickers)))
correlated_rv = np.transpose(np.matmul(chol_mat, np.transpose(rv)))

u = np.mean(returns, axis=0).values
sigma = np.std(returns, axis=0).values
S_0 = adj_close.values[-1, :]
P_0 = np.sum(shares * S_0)
st.text('Initial value')
st.text(P_0)

wt=np.sqrt(t) * correlated_rv
S_T = S_0 * np.exp((u - 0.5 * sigma ** 2) * t + sigma * wt)
#st.dataframe(S_T)

P_T = np.sum(shares * S_T, axis=1)
P_diff = P_T - P_0
P_diff

P_diff_sorted = np.sort(P_diff)
percentiles = [0.1, 1., 5.]
var = np.percentile(P_diff_sorted, percentiles)
for x, y in zip(percentiles, var):
    st.text(f'1-day VaR with {100-x}% confidence: {-y:.2f}$')
    

ax = sns.histplot(P_diff, kde=False)
ax.set_title('''Distribution of possible 1-day changes in portfolio value 1-day''', fontsize=16)
ax.axvline(var[0], 0, 10000).set_color('red');
ax.axvline(var[1], 0, 10000).set_color('orange');
ax.axvline(var[2], 0, 10000).set_color('yellow');
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.title('Expected shortfall ')
var999 = np.percentile(P_diff_sorted, 0)
expected_shortfall999 = P_diff_sorted[P_diff_sorted<=var999].mean()
var99 = np.percentile(P_diff_sorted, 1)
expected_shortfall99 = P_diff_sorted[P_diff_sorted<=var99].mean()
var95 = np.percentile(P_diff_sorted, 2)
expected_shortfall95 = P_diff_sorted[P_diff_sorted<=var95].mean()
st.text('Expected shortfall 99,9%: ' + str(round(expected_shortfall999, 2))+'$')
st.text('Expected shortfall 99,0%: ' + str(round(expected_shortfall99, 2))+'$')
st.text('Expected shortfall 95,0%: ' + str(round(expected_shortfall95, 2))+'$')


#fatto con mediana
st.title('Median Expected shortfall ')
mvar999 = np.percentile(P_diff_sorted, 0)
mexpected_shortfall999 = P_diff_sorted[P_diff_sorted<=mvar999]
mexpected_shortfall999 = np.median(mexpected_shortfall999)
mvar99 = np.percentile(P_diff_sorted, 1)
mexpected_shortfall99 = P_diff_sorted[P_diff_sorted<=mvar99]
mexpected_shortfall99 = np.median(mexpected_shortfall99)
mvar95 = np.percentile(P_diff_sorted, 2)
mexpected_shortfall95 = P_diff_sorted[P_diff_sorted<=mvar95]
mexpected_shortfall95 = np.median(mexpected_shortfall95)
st.text('Expected shortfall 99,9%  ' + str(round(mexpected_shortfall999,2))+'$')
st.text('Expected shortfall 99,0%  ' + str (round(mexpected_shortfall99,2))+'$')
st.text('Expected shortfall 95,0%  ' + str (round(mexpected_shortfall95,2))+'$')

st.title('Percentual Expected shortfall ')
pexpected_shortfall999=expected_shortfall999/P_0
pexpected_shortfall99=expected_shortfall99/P_0
pexpected_shortfall95=expected_shortfall95/P_0
st.text('Percentual expected shortfall 99,9%  ' + str (round(pexpected_shortfall999*100,2))+'%')
st.text('Percentual expected shortfall 99,0%  ' + str (round(pexpected_shortfall99*100,2))+'%')
st.text('Percentual expected shortfall 95,0%  ' + str (round(pexpected_shortfall95*100,2))+'%')

var_array = []
num_days = int(15)
for x in range(1, num_days+1):    
    var_array.append(np.round(var * np.sqrt(x),2))
vardays=pd.DataFrame(var_array,index = range(1,num_days+1))
vardays.columns = ["VaR 99.9", "VaR 99", "VaR 95"]
st.write(vardays.round(0).transpose())


pvardays=vardays/P_0*100
pvardays.columns = ["% VaR 99.9", "% VaR 99", "% VaR 95"]
st.write(pvardays.round(2).transpose())

plt.xlabel("Day #")
plt.ylabel("Max portfolio loss (USD)")
plt.title("Max portfolio loss (VaR) 15-day period")
plt.plot(vardays)
plt.rcParams["figure.figsize"] = [10,5]
st.line_chart(vardays)
