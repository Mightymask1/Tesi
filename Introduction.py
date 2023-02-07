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

st.sidebar.markdown('---')
st.sidebar.write('Developed by Pietro Daloiso')
