# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:46:32 2021

@author: patta
"""
# imports 
import numpy as np
import pandas as pd
import os

# # check path
os.chdir(r"C:\Users\patta\Sustainability Analysis in Python\Assignment\Sustainability-Analysis-in-Python")
print("Current Working Directory " , os.getcwd())

# read data
df = pd.read_csv('Goal1.csv')

#%% filter the data
# drop nan columns
df = df.drop(['TimeCoverage', 'UpperBound', 'LowerBound','BasePeriod','GeoInfoUrl','FootNote'], axis=1)

#country list
countries = df['GeoAreaName'].to_list()
unique_countries = pd.unique(countries)

#%% sanity checks for each country
for country in unique_countries:
    subset = df[df['GeoAreaName'] == country]
    minimum=subset['Value'].min()
    maximum=subset['Value'].max()
    avg=subset['Value'].mean()
    print(country," : ", 
          'The minimum value is', minimum,
          'The maximum value is', maximum,
          'The average value is', avg)
    



          

