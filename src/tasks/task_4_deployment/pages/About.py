import streamlit as st

st.set_page_config(
    page_title='Weather Prediction',
    page_icon=':sun_behind_cloud',
    initial_sidebar_state='auto',
)

st.title('Karachi Pakistan Weather Prediction', anchor=None)
st.subheader('About this project')

st.markdown(
"""
The goal of this project is to develop a machine learning model that can improve the accuracy of weather forecasts for Pakistan. The model will be trained on a dataset of historical weather data, and it will be able to predict future weather conditions with greater accuracy than current models.

While scoping the project, it was decided to focus on Karachi but the result of the model should be reusable for other cities.

*for more information on the initial project:*

[Omdena | Advancing weather prediction with machine learning and python](https://omdena.com/chapter-challenges/advancing-weather-prediction-with-machine-learning-and-python/)
"""
)