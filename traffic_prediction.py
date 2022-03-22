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

st.title('Kaggle-Project: EDA on Traffic Prediction')

header = st.container()
data = st.container()
features = st.container()
insight_1 = st.container()
insight_2 = st.container()

with header:
    st.subheader("Description:")
    st.text("For the March edition of the 2022 Tabular Playground Series you're challenged to \nforecast twelve-hours of traffic flow in a U.S. metropolis.\
            \nThe time series in this dataset are labelled with both location coordinates and\na direction of travel -- a combination of features that will test\
            \nyour skill at spatio-temporal forecasting within a highly dynamic traffic network. \nWhich model will prevail? The venerable linear regression?\
            \nThe deservedly-popular ensemble of decision trees? Or maybe a cutting-edge\ngraph neural-network? We can't wait to see!")


with data:
    st.title("Our Dataset after Feature Engineering:")
    df = pd.read_csv('fe_modified.csv')
    df = df.reindex(columns=['time','year','month','date','hour','minute','is_weekday','m_a_e_n','direction',
                            'road_coord','highway_code','congestion'])
    if st.button('Show data'):
        st.dataframe(df.head(300))
    
    st.write('Description of new Categorical feature columns')
    st.write('is_weekday =[0-Weekday, 1-Weekend]')
    st.write('m_a_e_n is a new feature that encode hours column')
    st.write('5-11--Morning')
    st.write('12-16--afternoon')
    st.write('17-21--evening')
    st.write('22-04--night')

 

with insight_1: 
    st.title('Insights')
    st.text('Looking through the Data we got some intersting Insights:')
    ## Distrubution of data 
    st.subheader('1.The aim of this distrubution is to see which range of congestion has more datapoints')
    fig = px.histogram(df, x='congestion', nbins=100, title='Data Distrubution over congestion')
    plt.xlabel('Congestion', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    st.plotly_chart(fig)
    st.subheader('The finding was more data points lies in range of 30-70')

###################################
    st.subheader("2.Visualizing to find any month has much higher traffic congestion")
    plt.figure(figsize=(14,7))
    df1 = df.groupby(['month'], as_index=False)['congestion'].mean()

    fig = px.bar(df1, df1['month'],df1['congestion'], color='month', title='Mean congestion per month')
    fig.update(layout_yaxis_range=[45,50])

    st.plotly_chart(fig)
    st.subheader('It seems the traffic congestion is much more the same in all months')
#########################################
    st.subheader("3.To check wether weekdays and weekends have any effect on traffic congestion")
    ## traffic congestion 
    df2 = df.groupby(['is_weekday'], as_index=False)['congestion'].mean()
    fig = px.bar(df2, x=['weekday','weekend'], y=df2['congestion'], color = 'is_weekday',title='Mean congestion over weekdays and weekends' )
    fig.update(layout_yaxis_range=[45,50])


    st.plotly_chart(fig)
    st.subheader('It seems the traffic congestion in weekends is bit lower.')
##############################################
    ## traffic congestion over hour
    st.subheader('4.To find is there any Rush hours')
    df3 = df.groupby(['hour'], as_index=False)['congestion'].mean()
    fig = px.line(df, df3.hour, df3['congestion'], title='Patren of traffic congestion by every hour')
    fig.update_layout(xaxis_title="Hours(0-24)", yaxis_title="Congestion(mean)")
    st.plotly_chart(fig)
    st.subheader('There was a peak during 12-17 hours.')

#########################################################33
    st.subheader("5.To check  traffic congestion during day")
    ## traffic congestion 
    df3 = df.groupby(['m_a_e_n'], as_index=False)['congestion'].mean()
    fig = px.bar(df3, x=['morning','afternoon','evening','night'], y=df3['congestion'], color = 'm_a_e_n',title='Traffic Congestion for throughout the day' )
    fig.update(layout_yaxis_range=[35,55])
    fig.update_layout(xaxis_title="Throughout the day", yaxis_title="Congestion(mean)")

    st.plotly_chart(fig)
    st.subheader('It seems the traffic congestion during morning(05-11 hours) and night(22-04 hours) is bit low')
#####################################################
    st.subheader('6.To check directions have any impact on Traffic congestion')
    fig = plt.figure(figsize=(10, 4))

    sns.barplot(x='direction',y='congestion', data=df)
    plt.title('Impact of Roadway direction on congestion')
    st.pyplot(fig)
    st.subheader("The more traffic is in EB, NB, SB, WB")
    

