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

with open('config.json', 'r') as f:
    # Load JSON data from file
    json_data = json.load(f)

# Load evo_model
with open(json_data['evo_model'], 'rb') as file:
    evo_model = pickle.load(file)

# Load pre_rate_model
with open(json_data['pre_rate_model'], 'rb') as file:
    pre_rate_model = pickle.load(file)

print(evo_model)


data_columns = json_data['data_columns']
now = datetime.now() - relativedelta(days=7)
start = now - relativedelta(months=11)
date_string_end = now.strftime('%Y-%m-%d')
date_string_start = start.strftime('%Y-%m-%d')
date_pred = []
for date in pd.date_range(start=datetime.now() - relativedelta(days=6), periods=10):
    date_pred.append(date.strftime('%Y-%m-%d'))


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
##need max temp, evotranspiration, precipitation_rate
print(concat_df.columns)
total_hours = concat_df['precipitation_hours'].sum()
concat_df['precipitation_rate'] = concat_df['precipitation_sum']/total_hours

##generate prediction for max temp
max_temp = concat_df['temperature_2m_max'].values
# Define and fit the model
max_temp_model = ARIMA(max_temp, order=(5,1,0))
max_temp_model = max_temp_model.fit()
# Make predictions
max_temp_predictions = max_temp_model.predict(start=1, end=10)
print(len(max_temp_predictions))


##generate prediction for evo_transpiration
et0_fao_evapotranspiration = TimeSeries.from_series(concat_df['et0_fao_evapotranspiration'].values)
scaler = StandardScaler()
transformer = Scaler(scaler)
series_transformed = transformer.fit_transform(et0_fao_evapotranspiration)
evo_model.fit(
    series=series_transformed, verbose=0,
          )
evo_preds = evo_model.predict(n=10, series=series_transformed)
evo_preds = transformer.inverse_transform(evo_preds)
print(len(evo_preds))
print(evo_preds)

##generate prediction for precipitation rate
precipitation_rate = TimeSeries.from_series(concat_df['precipitation_rate'].values)
scaler = StandardScaler()
transformer = Scaler(scaler)
series_transformed = transformer.fit_transform(precipitation_rate)
pre_rate_model.fit(
    series=series_transformed, verbose=0,
          )
precipitation_rate_preds = pre_rate_model.predict(n=10, series=series_transformed)
precipitation_rate_preds = transformer.inverse_transform(precipitation_rate_preds)
print(len(precipitation_rate_preds))
print(precipitation_rate_preds)

## Visual Element
# headerImage = Image.open('OmdenaHeaderImage.jpg')

##Streamlit app
st.set_page_config(page_title="Weather Prediction", page_icon="⛅")
# st.image(headerImage)
st.title('Karachi Pakistan Weather Prediction')
if concat_df is not None:
    display_df = concat_df[['temperature_2m_max', 'et0_fao_evapotranspiration', 'precipitation_rate']]
    st.write("Current values")
    st.write(display_df.tail(20))

    st.write("Predicted future values - To be implemented")

    # Plot the first column
    st.subheader(f'Plot of temperature_2m_max over time')
    fig1, ax1 = plt.subplots()
    ax1.plot(concat_df.index, concat_df['temperature_2m_max'])
    # Set the locator
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    st.pyplot(fig1)

    # Plot the second column
    st.subheader(f'Plot of et0_fao_evapotranspiration over time')
    fig2, ax2 = plt.subplots()
    ax2.plot(concat_df.index, concat_df['et0_fao_evapotranspiration'])
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    st.pyplot(fig2)

    # Plot the third column
    st.subheader(f'Plot of precipitation_rate over time')
    fig3, ax3 = plt.subplots()
    ax3.plot(concat_df.index, concat_df['precipitation_rate'])
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    st.pyplot(fig3)
