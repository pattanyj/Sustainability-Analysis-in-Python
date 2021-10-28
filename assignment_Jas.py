# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:46:32 2021

@author: patta
"""
#%%
# imports 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from assignment_B_model import logistic, calibration, nmrse

# check path
os.chdir(r"C:\Users\patta\Sustainability Analysis in Python\Assignment\Sustainability-Analysis-in-Python")
print("Current Working Directory " , os.getcwd())

# read data
df = pd.read_csv('Goal1.csv')

#%% filter the data
# drop nan columns
df = df.drop(['Goal','Target','GeoAreaCode','Source','Nature','Reporting Type','TimeCoverage','UpperBound', 'LowerBound','BasePeriod','GeoInfoUrl','FootNote'], axis=1)
df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]

# focusing on women between 15-24 only 
df_female = df.loc[df["Sex"]=="FEMALE"]
df_final = df_female.loc[df_female["Age"]=="15-24"]

#country list
unique_countries = pd.unique(df_final['GeoAreaName'].to_list())

#%% sanity checks for each country
for country in unique_countries:
    subset = df_final[df_final['GeoAreaName'] == country]
    minimum=subset['Value'].min()
    maximum=subset['Value'].max()
    avg=subset['Value'].mean()
    # print(country," : ", 
    #       'The minimum value is', minimum,
    #       'The maximum value is', maximum,
    #       'The average value is', avg)
    
#%% main analysis

# Using calibration and logistic functions
popt_list=[]
logistic_values=[]

for country in unique_countries:
    subset = df_final[df_final['GeoAreaName'] == country]
    year = subset['TimePeriod']
    value = subset['Value']
    temp_popt = calibration(year, value)
    popt_list.append(temp_popt)
    log_list = logistic(year,temp_popt[0],temp_popt[1],temp_popt[2],temp_popt[3])
    logistic_values.append(log_list)

# Evaluate the model using R2
y_true = df_final['Value'].to_list()

# unpack the y_pred values from the log function with a nested for loop
y_pred = []
for value in logistic_values:
    for item in value:
        y_pred.append(item)
 
#calculate r2
r2_value = r2_score(y_true, y_pred)

#NRMSE
#calculate RMSE
rmse=mean_squared_error(y_true, y_pred)
avg_y_true = sum(y_true)/len(y_true)
nrmse = rmse/avg_y_true

#PBIAS
#create new df to seperate by country
data ={'Countries':df_final['GeoAreaName'], 'Years':df_final['TimePeriod'],  'Y_true':y_true,'Y_pred':y_pred}
pbias_df = pd.DataFrame(data)
# pbias_df=pbias_df.groupby(Countries)      #this doesnt work

pbias_values=[]
for country in unique_countries:
    subset = pbias_df[pbias_df['Countries'] == country]
    y_true_1 = subset['Y_true']
    y_pred_1 = subset['Y_pred']
    y_true_1 = np.array(y_true_1)
    y_pred_1 = np.array(y_pred_1)
    pbias = 100* ((sum(y_pred_1-y_true_1))/sum(y_pred_1))
    pbias_values.append(pbias)

#%% Validate using 5-fold cross validation

# logistic validation
validated_data=pd.DataFrame(data=unique_countries, columns=['Country'])
calibration_results=[]
x_test=[]

for location in range(len(unique_countries)):
    country=unique_countries[location]
    data=df_final[:][df_final.GeoAreaName==country]
    x=data.TimePeriod.to_numpy()
    y=data.Value.to_numpy()
    kf_5 = KFold(n_splits=min(5, len(y)))   # takes the minimum for when there are less than 5 data points
    for train_index, test_index in kf_5.split(x):
          X_train, X_test = x[train_index], x[test_index]
          y_train, y_test = y[train_index], y[test_index]
    calibration_results.append(calibration(X_train, y_train))
    x_test.append(X_test)
    
validated_data=pd.concat([validated_data, pd.DataFrame(calibration_results)], axis=1)
validated_data.columns =['Country','start','K', 'x_peak', 'r']

y_simulated=np.array([])
country_log=np.array([])
for location in range(len(unique_countries)):
    country=unique_countries[location]
    x_data=x_test[location]
    K = validated_data['K'].loc[validated_data['Country']==country]
    r= validated_data['r'].loc[validated_data['Country']==country]
    s=validated_data['start'].loc[validated_data['Country']==country]
    x_p=validated_data['x_peak'].loc[validated_data['Country']==country]
    for x_point in x_data:
        y_simulated=np.append(y_simulated,logistic(x_point,s,K,x_p,r))
        country_log=np.append(country_log,country)

data_sim = {'Country':country_log,  'Simulationed Y': y_simulated}
logistic_validation  = pd.DataFrame(data=data_sim)

#%%Calculating the R2 per country
r2=[]
pair=[]
observed=[]

for location in range(len(unique_countries)):
    country=unique_countries[location]
    years=x_test[location]
    for year in years:
        pair.append(np.where((df_final['GeoAreaName']==country)
                            &(df_final['TimePeriod']==year)))
    for pair in pair:
        observed.append(df_final.Value.iloc[pair].values)
    simulated=logistic_validation['Simulationed Y'][logistic_validation['Country']==country].tolist()
    r2.append(mean_squared_error(observed, simulated))
    observed=[]
    pair=[]

# calculate nrmse 
for location in range(len(unique_countries)):
    country=unique_countries[location]
    years=x_test[location]
    for year in years:
        list_.append(np.where((df_final['Value']==country)
                            &(df_final['TimePeriod']==year)))

nrmse_df_validation = logistic_validation
nrmse_df_validation['Real Y'] = 
nmrse_validation=[]

# for country in unique_countries:
#     subset = df_final[df_final['Countries'] == country]
#     country=unique_countries[location]

        


#%%
'wrong'
# X = np.array(pbias_df['Years'])
# y = np.array(pbias_df['Y_true'])
# kf5 = KFold(n_splits=5)
# kf10 = KFold(n_splits=10)
# kf5.get_n_splits(X)

# # use X_testing data for calibration and logistic model - ytest= values and x=test years  

# logistic_validation = []

# for train_index, test_index in kf5.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index] 
#     ctest_validation = calibration(X_test,y_test)
#     logtest_validation = logistic(y_test,ctest_validation[0],ctest_validation[1],ctest_validation[2],
#                     ctest_validation[3])
#     logistic_validation.append(logtest_validation)





    #%%
# data ={'X test':X_test, 'Y test':y_test}
# test_df = pd.DataFrame(data)

# # run the test values through the calibration and logistic model
# ctest_validation = calibration(X_test,y_test)
# logtest_validation = logistic(y_test,ctest_validation[0],ctest_validation[1],ctest_validation[2],
#                     ctest_validation[3])

# # run the train values through the calibration and logistic model
# ctrain_validation = calibration(X_train,y_train)
# logtrain_validation = logistic(X_train,ctest_validation[0],ctest_validation[1],ctest_validation[2],
#                     ctest_validation[3])
# # for item in test_df.T.iteritems():

    
#%% validation with R2
r2_validation_train = r2_score(X_train, y_train)
r2_validation_test = r2_score(X_test, y_test)

# validation with NRMSE
rmse=mean_squared_error(X_train, y_train)
avg_y_true = sum(X_train)/len(X_train)
nrmse_validation = rmse/avg_y_true

#%% Button

'Still working on this, it runs but not connected to code'

import sys
from PyQt5.QtWidgets import (QLabel, QRadioButton, QPushButton, QVBoxLayout, QApplication, QWidget)


class basicRadiobuttonExample(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.label = QLabel('Select the cross-validation method')
        self.rbtn1 = QRadioButton('5-fold cross validation')
        self.rbtn2 = QRadioButton('10-fold cross validation')
        self.label2 = QLabel("")
        
        # self.rbtn1.toggled.connect(self.onClicked)
        # self.rbtn2.toggled.connect(self.onClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.rbtn1)
        layout.addWidget(self.rbtn2)
        layout.addWidget(self.label2)
        
        self.setGeometry(600, 600,900, 50)
        # self.setFont(QFont('Arial', 10)
        self.setLayout(layout)
        self.setWindowTitle('Input Needed')
        

        self.show()

    # def onClicked(self):
    #     radioBtn = self.sender()
    #     if radioBtn.isChecked():
    #         self.label2.setText("You live in " + radioBtn.text())


if __name__ == '__main__':    
    app = QApplication(sys.argv)
    ex = basicRadiobuttonExample()
    sys.exit(app.exec_())