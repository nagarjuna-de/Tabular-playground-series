import streamlit as st
import pandas as pd
import numpy as np
from operator import index
import re
from turtle import width
import streamlit as st
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.title('Kaggle-Project: Traffic Prediction')

header = st.container()
data = st.container()
features = st.container()
insight_1 = st.container()
insight_2 = st.container()

with header:
    st.title("Exercise:")
    st.text("For the March edition of the 2022 Tabular Playground Series you're challenged to \nforecast twelve-hours of traffic flow in a U.S. metropolis.\
            \nThe time series in this dataset are labelled with both location coordinates and\na direction of travel -- a combination of features that will test\
            \nyour skill at spatio-temporal forecasting within a highly dynamic traffic network. \nWhich model will prevail? The venerable linear regression?\
            \nThe deservedly-popular ensemble of decision trees? Or maybe a cutting-edge\ngraph neural-network? We can't wait to see!")


with data:
    st.title("Our Dataset after classification:")
    data = pd.read_csv('fe_modified.csv')
    st.write(data.head(10))

 

with insight_1: 
    df = pd.read_csv('fe_modified.csv')
    st.title('Insights')
    st.text('Looking through the Data we got some intersting Insights:')
    fig = px.histogram(df, x='congestion', nbins=100, title='Data Distrubution over congestion')
    plt.xlabel('Congestion', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    st.plotly_chart(fig)
    
    plt.figure(figsize=(14,7))
    df1 = df.groupby(['month'], as_index=False)['congestion'].mean()
    
    fig = px.bar(df1, df1['month'],df1['congestion'], color='month', title='Mean congestion per month')
    fig.update(layout_yaxis_range=[45,50])
    #plt.title(f'Mean congestion per month', fontsize=16)
    plt.xlabel('Month-number', fontsize=16)
    plt.ylabel('Congestion', fontsize=16)

with insight_2:
    df = pd.read_csv('fe_modified.csv')
    