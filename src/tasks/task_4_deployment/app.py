import requests
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import matplotlib.pyplot as plt

with open('config.json', 'r') as f:
    # Load JSON data from file
    json_data = json.load(f)

data_columns = json_data['data_columns']
now = datetime.now() - relativedelta(days=7)
start = now - relativedelta(years=1)
date_string_end = now.strftime('%Y-%m-%d')
date_string_start = start.strftime('%Y-%m-%d')

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

##Streamlit app
st.title('Karachi Pakistan Weather Prediction')
if concat_df is not None:
    st.write(concat_df)

    # Plot the first column
    st.subheader(f'Plot of temperature_2m_max over time')
    fig1, ax1 = plt.subplots()
    ax1.plot(concat_df.index, concat_df['temperature_2m_max'])
    st.pyplot(fig1)

    # Plot the second column
    st.subheader(f'Plot of et0_fao_evapotranspiration over time')
    fig2, ax2 = plt.subplots()
    ax2.plot(concat_df.index, concat_df['et0_fao_evapotranspiration'])
    st.pyplot(fig2)

    # Plot the third column
    st.subheader(f'Plot of precipitation_rate over time')
    fig3, ax3 = plt.subplots()
    ax3.plot(concat_df.index, concat_df['precipitation_rate'])
    st.pyplot(fig3)






