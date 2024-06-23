# -*- coding: utf-8 -*-
"""Optimization_of_Electrical_Consumption_Data_Preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VoqouCgb2BG4mAOGg9WZuKmln5p09YY2
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
#importing libraries

import pandas as pd          # data analysis library for handling structured data
import numpy as np           # mathematical library for working with numerical data
import datetime

# Visualization
import matplotlib.pyplot as plt     # data visualization library for creating graphs and charts
# %matplotlib inline
import seaborn as sns        # data visualization library based on matplotlib for creating more attractive visualizations
import plotly
import plotly.express as px   # interactive data visualization library
import plotly.graph_objects as go   # library for creating interactive graphs and charts
from plotly.subplots import make_subplots
import missingno as msno

#/content/drive/MyDrive/Data/HomeC.csv

smart_home = pd.read_csv('/content/drive/MyDrive/Data/HomeC.csv')

smart_home['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(smart_home),  freq='min'))

smart_home['year'] = smart_home['time'].apply(lambda x : x.year)
smart_home['month'] = smart_home['time'].apply(lambda x : x.month)
smart_home['day'] = smart_home['time'].apply(lambda x : x.day)
smart_home['weekday'] = smart_home['time'].apply(lambda x : x.day_name())
smart_home['weekofyear'] = smart_home['time'].apply(lambda x : x.weekofyear)
smart_home['hour'] = smart_home['time'].apply(lambda x : x.hour)
smart_home['minute'] = smart_home['time'].apply(lambda x : x.minute)

def hours2timing(x):
    if x in [20,21,22,23,0,1,2,3,4]:
        timing = 'Night'
    elif x in range(4, 12):
        timing = 'Morning'
    elif x in range(12, 16):
        timing = 'Afternoon'
    elif x in range(16, 20):
        timing = 'Evening'
    else:
        timing = 'X'
    return timing

smart_home['timing'] = smart_home['hour'].apply(hours2timing)

smart_home.columns = [i.replace(' [kW]', '') for i in smart_home.columns]

smart_home['Furnace'] = smart_home[['Furnace 1','Furnace 2']].sum(axis=1)
smart_home['Kitchen'] = smart_home[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1)
smart_home.drop(['Furnace 1','Furnace 2','Kitchen 12','Kitchen 14','Kitchen 38','icon','summary'], axis=1, inplace=True)

smart_home[smart_home.isnull().any(axis=1)]

smart_home = smart_home[0:-1]

from sklearn.covariance import EllipticEnvelope

# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)

# Select only numeric columns
numeric_cols = smart_home.select_dtypes(include=["float64", "int64"])

# Fit detector
outlier_detector.fit(numeric_cols)

# Predict outliers
outlier_predictions = outlier_detector.predict(numeric_cols)

# Add outlier predictions to original dataframe
smart_home["outlier"] = outlier_predictions




# The EllipticEnvelope algorithm is a robust covariance estimator that fits an ellipse to the central region of the data,
# ignoring observations that are considered outliers. It is a method for detecting outliers in multivariate data that are
# assumed to be normally distributed. The contamination parameter specifies the proportion of data points that are expected
# to be outliers.

# The predict method of the LOF object is then used to predict whether each observation in numeric_cols is an outlier
# or not, with 1 indicating a normal observation and -1 indicating an outlier

#Filtering the data into Overall Energy consumption and generation, energy consumption of diifferent appliances
#Remaining data considered as weather data
#select the required columns
smart_home['overall_energy_consumption'] = smart_home['use']
smart_home['overall_energy_generation'] = smart_home['gen']

# Save the updated DataFrame to a CSV file
output_csv_path = '/content/drive/MyDrive/Data/smart_home_data_preprocessed.csv'
smart_home.to_csv(output_csv_path)