"""
Freely adapted from another Omdena app:
https://saudi-arabia-industrial-co2.streamlit.app
"""

# Import of all required libraries
import requests
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import statsmodels.api as sm
import darts
import random
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from darts.models import RegressionModel, CatBoostModel, RandomForest, LightGBMModel, XGBModel, RNNModel
from darts.metrics import rmse, mape
from statsmodels.tsa.arima.model import ARIMA
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Set the work directory to retrieve all data
# script_dir = os.path.dirname(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))

config_json = "config.json"
models_dir = "models"
rel_to_config_json_path = os.path.join(script_dir, config_json)

with open(rel_to_config_json_path, 'r') as f:
    # Load JSON data from file
    json_data = json.load(f)

# Load evo_model json_data['evo_model']
evo_model_path = os.path.join(script_dir, models_dir, json_data["evo_model"])
evo_model = RNNModel.load(evo_model_path, map_location='cpu')
evo_model.to_cpu()


# Load pre_rate_model
pre_rate_path = os.path.join(script_dir, models_dir, json_data["pre_rate_model"])
pre_rate_model = RNNModel.load(pre_rate_path, map_location='cpu')
pre_rate_model.to_cpu()

# Format to datetime
data_columns = json_data['data_columns']
now = datetime.now() - relativedelta(days=7)
start = now - relativedelta(months=11)
date_string_end = now.strftime('%Y-%m-%d')
date_string_start = start.strftime('%Y-%m-%d')
date_pred = []
for date in pd.date_range(start=datetime.now() - relativedelta(days=6), periods=10):
    date_pred.append(date.strftime('%Y-%m-%d'))

# Plug to live API to retrive live data
url = "https://archive-api.open-meteo.com/v1/archive"
cities = [
    { "name": "Karachi", "country": "Pakistan", "latitude": 24.8608, "longitude": 67.0104 }
]
cities_df =[]
for city in cities:
    params = {"latitude":city["latitude"],
            "longitude":city['longitude'],
            "start_date": date_string_start,
            "end_date": date_string_end,
            "daily": data_columns,
            "timezone": "GMT",
            "min": date_string_start,
            "max": date_string_end,
    }
    res = requests.get(url, params=params)
    data = res.json()
    df = pd.DataFrame(data["daily"])
    df["latitude"] = data["latitude"]
    df["longitude"] = data["longitude"]
    df["elevation"] = data["elevation"]
    df["country"] = city["country"]
    df["city"] = city["name"]
    cities_df.append(df)
concat_df = pd.concat(cities_df, ignore_index=True)
concat_df.set_index('time', inplace=True)
print(concat_df.columns)
total_hours = concat_df['precipitation_hours'].sum()
concat_df['precipitation_rate'] = concat_df['precipitation_sum']/total_hours

# generate prediction for max temp
max_temp = concat_df['temperature_2m_max'].values
# Define and fit the model
max_temp_model = ARIMA(max_temp, order=(5,1,0))
max_temp_model = max_temp_model.fit()
# Make predictions
max_temp_predictions = max_temp_model.predict(start=1, end=10)

# generate prediction for evo_transpiration
mean_evo = concat_df['et0_fao_evapotranspiration'].mean()
et0_fao_evapotranspiration = TimeSeries.from_series(concat_df['et0_fao_evapotranspiration'].values, fillna_value=mean_evo)
scaler = StandardScaler()
transformer = Scaler(scaler)
series_transformed = transformer.fit_transform(et0_fao_evapotranspiration)
evo_model.fit(series=series_transformed, verbose=0)
print(f'MODEL: {evo_model}')
evo_preds = evo_model.predict(10, series=series_transformed)
evo_preds = transformer.inverse_transform(evo_preds).univariate_values()
print(evo_preds)
print(len(evo_preds))

print('--------------------')
# generate prediction for precipitation rate
#mean_pre = concat_df['et0_fao_evapotranspiration'].mean()
precipitation_rate = TimeSeries.from_series(concat_df['precipitation_rate'].values, fillna_value=0)
scaler = StandardScaler()
transformer = Scaler(scaler)
series_transformed = transformer.fit_transform(precipitation_rate)
pre_rate_model.fit(
    series=series_transformed, verbose=0,
          )
precipitation_rate_preds = pre_rate_model.predict(n=10, series=series_transformed)
precipitation_rate_preds = transformer.inverse_transform(precipitation_rate_preds).univariate_values()
print(precipitation_rate_preds)
print(len(precipitation_rate_preds))

## make future value df
predicted_data = {'time': date_pred, 'temperature_2m_max': max_temp_predictions, 'et0_fao_evapotranspiration': evo_preds, 'precipitation_rate': precipitation_rate_preds}
# Create Future value DataFrame
predicted_df = pd.DataFrame(predicted_data)
predicted_df.set_index('time', inplace=True)
print(predicted_df)

## Streamlit app

# Visual Elements
color_palette = {
    'temperature_2m_max': '#EB5858' ,
    'et0_fao_evapotranspiration': '#FFE760',
    'precipitation_rate': '#85DBF7'
}

color_palette_pred = {
    'temperature_2m_max': '#5887EB' ,
    'et0_fao_evapotranspiration': '#0074FF',
    'precipitation_rate': '#FFA500'
}

labels_dict = {
    'temperature_2m_max': 'Max Temperature (Celsius)' ,
    'et0_fao_evapotranspiration': 'Evapotranspiration (millimeters per day)',
    'precipitation_rate': 'Precipitation rate (millimeters per Hour)'
}

labels_english = {
    'temperature_2m_max': 'Maximum daily air temperature at 2 meters above ground (Celsius)' ,
    'et0_fao_evapotranspiration': 'ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.',
    'precipitation_rate': 'The daily su of precipitation divided by the number of hours with rain.'
}

# Page creation
st.set_page_config(page_title="Weather Prediction", page_icon="⛅")
st.title('Karachi Pakistan Weather Prediction')

st.subheader('Project Scope')
st.write("""
        Studying weather data from Karachi from 2010, this project aims at giving ten days prediction
         on features that are important for the catastrophy prevention:

         - Daily Max Temperature (Heatwave)

         - et0_fao_evapotranspiration and precipitation rate (flooding)

         Note: We plan to update that app to display warning in case one of our indicators is above a certain threshold.

         """)

# Warning in case data can't be uploaded)
if concat_df is None:
    st.warning('There is a problem with the data')
    st.stop()

# Select feature to predict
feature_list = ['temperature_2m_max', 'et0_fao_evapotranspiration', 'precipitation_rate']
display_df = concat_df[feature_list]

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: start;} </style>',
         unsafe_allow_html=True)
st.write('<style>.st-ec .st-df label {font-size: larger; font-weight: bold;}</style>', unsafe_allow_html=True)
forecast_choice = st.selectbox("Select the feature you want to forecast", (feature_list))



# Plot based on choice
st.subheader(f'Forecast')

# Create a layout with a title
layout = go.Layout(
    title=dict(
        text="",
        font=dict(
            family="Courier New, monospace",
            size=24,  # Font size for the title
            color="RebeccaPurple"
        )
    ),
    xaxis=dict(
        titlefont=dict(
            family="Arial, sans-serif",  # Different font family for x-axis title
            size=18  # Font size for x-axis title
         )
    ),
    yaxis=dict(title="Y Axis Title",
            titlefont=dict(
                family="Arial, sans-serif",  # Different font family for x-axis title
                size=16  # Font size for y-axis title
        ))
)

fig = go.Figure(layout=layout)
fig.update_xaxes(showline=True, linewidth=1, linecolor='rgb(96, 103, 117)', gridcolor='rgb(96, 103, 117)')
fig.update_yaxes(showline=True, linewidth=1, linecolor='rgb(96, 103, 117)', gridcolor='rgb(96, 103, 117)')
fig.update_layout(paper_bgcolor="#262730", plot_bgcolor="rgb(52, 53, 56)")

fig.add_trace(go.Scatter(x=concat_df.index,
                            y=concat_df[forecast_choice],
                            mode='lines',
                            marker_color=color_palette[forecast_choice],
                            name='Historical Value',
                            )
                )

fig.add_trace(go.Scatter(x=predicted_df.index,
                            y=predicted_df[forecast_choice],
                            mode='lines',
                            marker_color=color_palette_pred[forecast_choice],
                            name='Predicted Value'
                            )
                )

fig.update_yaxes(title_text=labels_dict[forecast_choice])
fig.update_layout(title=f"{forecast_choice} Forecasting for 10 years.")
st.plotly_chart(fig, theme=None)

with st.expander("Explanation"):
    st.image("OmdenaHeaderImage.jpg")
    st.write("""
        The chart above shows the prediction for the coming ten days in Karachi.
        It uses a Machine Learning model that have been trained on data from 2010 to 2023.
        ARIMA and LSTM have been used for that project.
             """)

display_df = concat_df[[forecast_choice]]
display_predicted_df = predicted_df[forecast_choice]

# Create the sidebar
st.sidebar.subheader(f'Features Description')
st.sidebar.write("""
                <div style="text-align: justify;">

                 **temperature_2m_max**:
                 
                 Maximum daily air temperature at 2 meters above ground (Celsius).
                 
                 **et0_fao_evapotranspiration**:
                 
                 ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
                 
                 **precipitation_rate**:
                 
                 The daily sum of precipitation divided by the number of hours with rain.
                </div>
                  """,
                unsafe_allow_html=True
                  )