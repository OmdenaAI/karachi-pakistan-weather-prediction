We collected the data from https://open-meteo.com/en/docs/historical-weather-api. We used requests and pandas library so as to collect. In this website, there are so many useful parameters for predicting the weather. 

weather code -> The most severe weather condition on a given day

temperature_2m_max, temperature_2m_min -> Maximum and minimum daily air temperature at 2 meters above ground

apparent_temperature_max, apparent_temperature_min ->Maximum and minimum daily apparent temperature = Heat Index

precipitation_sum -> Sum of daily precipitation (including rain, showers and snowfall)

rain_sum -> Sum of daily rain

snowfall_sum -> Sum of daily snowfall

precipitation_hours -> The number of hours with rain

sunrise, sunset -> Sun rise and set times

windspeed_10m_max, windgusts_10m_max -> Maximum wind speed and gusts on a day

winddirection_10m_dominant -> Dominant wind direction

shortwave_radiation_sum -> The sum of solar radiaion on a given day in cities

et0_fao_evapotranspiration -> Daily sum of ET₀ Reference Evapotranspiration of a well watered grass field

Unit

Temperature -> °C (°F)

precipitation_sum -> mm

rain_sum -> mm

snowfall_sum -> cm

precipitation_hours -> hours

Windspeed -> km/h

shortwave_radiation_sum -> MJ/m²

et0_fao_evapotranspiration -> mm
