import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

st.markdown(
    """
    <style>
    .app {
    background-color:#F5F5F5;
   font-family: "Times New Roman", Times, serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)
@st.cache_data
def get_data(filename):
    df = pd.read_csv(filename)
    return df


with header:
    st.title("Welcome to my Awesome Data Science Project!")
    st.text('In this project I  looked into the transaction of taxis in New York City')

with dataset:
    st.header('New York City Taxi Dataset')
    st.text('I found this dataset on kaggle.com')
    df = get_data('taxi_tripdata.csv')
    st.write(df.head())
    st.subheader('Pick-up location ID Distribution on the NYC Dataset')
    pulocation_dis = pd.DataFrame(df['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dis)



with features:
     st.header('The Features I created')
     st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
     st.markdown('* **second feature:** I created this feature because of this.. I calculated it using this logic..')

with modelTraining:
    st.header('Time to train the model')
    st.text('here you get to choose the hyperparameters of the model and see how the performance changes') 

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('what should be the max depth of the model?', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox('how many trees should there be?', options=[100, 200, 300, 'No Limit'], index=0)
    sel_col.text('Here is the List of Feature in my Data:')
    sel_col.write(df.columns)
    input_feature = sel_col.text_input('which feature should be used as input feature?','PULocationID')
   
    if n_estimators == "No Limit":
         regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    x = df[[input_feature]]
    y = df[['trip_distance']]
    
  

    regr.fit(x, y)
    prediction = regr.predict(x) 

    disp_col.subheader('Mean Absolute error of model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('Mean Squared Error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))
