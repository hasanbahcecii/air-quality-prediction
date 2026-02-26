"""
Air Quality Prediction System using LSTM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load csv data file
# values seperated with ";"
# use "," for decimal number in the data
df = pd.read_csv("AirQualityUCI.csv", sep= ";", decimal= ",", encoding= "latin1")
print(df.head())

# remove empty lines from the data
df.dropna(axis= 1, how= "all", inplace= True)
print(df.head())

# combine date and time columns 
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format= "%d/%m/%Y %H.%M.%S", errors= "coerce")
print(df.head())

# remove lines with errors in datetime
df.dropna(subset= ["datetime"], inplace= True)

# remove separate date and time columns because we combined them as datetime column
df.drop(["Date", "Time"], axis= 1, inplace= True)

# set index as datetime
df.set_index("datetime", inplace= True)
print(df.head())

# remove the rows containing incorrect sensor data (-200 means err in sensör data make it NaN)
df.replace(-200, np.nan, inplace= True)

# interpolation for null data
df.interpolate(method= "time", inplace= True)

# feature engineering
df["hour"] = df.index.hour
df["month"] = df.index.month

print(df.head())

# input and target variable (NO2(GT))
selected_columns = ["NO2(GT)", "T", "RH", "AH", "CO(GT)", "hour", "month"]
df = df[selected_columns]

# null value control
print(f"Null Value: {df.isnull().mean()*100}")

# korelation matrix 
plt.figure()
sns.heatmap(df.corr(), annot= True, fmt= ".2f", cmap= "YlGnBu")
plt.title("Korelation Matrix")
plt.show()

# target value time series graph
plt.figure()
df["NO2(GT)"].plot()
plt.title("NO2(GT) Time Series")
plt.xlabel("Date")
plt.ylabel("NO2(GT)")
plt.show()