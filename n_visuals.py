from operator import index
import re
from turtle import width
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#st.set_page_config(layout='wide')
# titles
#st.write('make your visuals here')
st.title('EDA on Traffic Congestion in 65-Roadways')
st.subheader("Insights on the given data set")

st.write('Feature Engineered Dataset')
df = pd.read_csv('fe_modified.csv')
df = df.reindex(columns=['time','year','month','date','hour','minute','is_weekday','m_a_e_n',
                         'road_coord','highway_code','congestion'])
if st.button('Show data'):
    st.dataframe(df.head(300))

## Distrubution of data 

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

st.plotly_chart(fig)