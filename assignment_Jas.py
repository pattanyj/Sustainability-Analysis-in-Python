# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:46:32 2021

@author: patta
"""
# imports 
import numpy as np
import pandas as pd
import os

# check path
os.chdir(r"C:\Users\patta\OneDrive\Desktop\Sustainability Analysis in Python\Assignment")
print("Current Working Directory " , os.getcwd())

# read data
df = pd.read_excel('Goal1.xlsx', sheet_name='data')

#%% filter the data
# print(df.isna().sum())
# drop nan columns
df = df.drop(['TimeCoverage', 'UpperBound', 'LowerBound','BasePeriod','GeoInfoUrl','FootNote'], axis=1)

#country list
countries = df['GeoAreaName'].to_list()
unique_countries = pd.unique(countries)

#calculate nan values for each country
for i in range(len(df['GeoAreaName'])):
    if unique_countries == df['GeoAreaName']:
        np.count(df['Value'])
        min(df['Value'])
        max(df['Value'])
        np.average(df['Value'])



          

