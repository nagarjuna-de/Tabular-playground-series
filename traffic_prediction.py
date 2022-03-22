import streamlit as st
import pandas as pd
import numpy as np

st.title('Kaggle-Project: Traffic Prediction')

header = st.container()
data = st.container()
features = st.container()
insights = st.container()

with header:
    st.title("Exercise:")
    st.text("For the March edition of the 2022 Tabular Playground Series you're challenged to \nforecast twelve-hours of traffic flow in a U.S. metropolis.\
            \nThe time series in this dataset are labelled with both location coordinates and\na direction of travel -- a combination of features that will test\
            \nyour skill at spatio-temporal forecasting within a highly dynamic traffic network. \nWhich model will prevail? The venerable linear regression?\
            \nThe deservedly-popular ensemble of decision trees? Or maybe a cutting-edge\ngraph neural-network? We can't wait to see!")


with data:
    st.title("Our Dataset after classification:")
    data = pd.read_csv('data/train.csv')
    st.write(data.head(10))

 

with features:
    st.title('features_title')

with insights: 
    st.title('Insights')
    st.text('Looking through the Data we got some intersting Insights:')