# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:46:32 2021

@author: patta

NOTES TO SELF:
    1. combine analysis for the task 4 steps r2,nmrse,pbias
    2. try to work out the next part similarly
    
"""
#%%
# check path 
'DELETE BEFORE SUBMISSION!!!!!!!'
import os
os.chdir(r"C:\Users\patta\Sustainability Analysis in Python\Assignment\Sustainability-Analysis-in-Python")
# print("Current Working Directory " , os.getcwd())

# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from assignment_B_model import logistic, calibration, nrmse, pbias, warn
import seaborn as sns
import warnings


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

y_values_df = pd.DataFrame({'Country': df_final['GeoAreaName'], 'Y_true': df_final['Value'], 'Y_predicted': y_pred})

#R2
r2_values=[]
#NRMSE
nrmse_values=[]
#PBIAS
pbias_values=[]
#remove warnings from console
warnings.warn = warn

for country in unique_countries:
    subset = df_final[df_final['GeoAreaName'] == country]
    y_true = subset['Value'].to_list()
    subset2 = y_values_df[y_values_df['Country'] == country]
    y_predicted = subset2['Y_predicted'].to_list()
    
    #R2
    r2_value = r2_score(y_true, y_predicted)
    r2_values.append(r2_value)
    
    #nrmse
    nrmse_value = nrmse(y_true, y_predicted)
    nrmse_values.append(nrmse_value)
    
    #PBIAS
    pbias_value = pbias(y_true,y_predicted)
    pbias_values.append(pbias_value)

evaluation_df = pd.DataFrame({'Countries': unique_countries, 'r2': r2_values, 'nrmse': nrmse_values, 'pbias': pbias_values})
# evaluation_df.to_csv(r'Evaluation.csv')

# Validate using 5-fold cross validation
simulated_values = []
#R2
r2_validation=[]
#NRMSE
nrmse_validation=[]
#PBIAS
pbias_validation=[]

k_value = 5 #change this value

for country in unique_countries:
    subset = df_final[df_final['GeoAreaName'] == country]
    x = subset['TimePeriod'].to_numpy()
    y = subset['Value'].to_numpy()
    kf = KFold(n_splits=min(k_value, len(y)))   # takes the minimum for when there are less than k_value data points
    
    r2_country=[]
    nrmse_country=[]
    pbias_country=[]
    
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
   
        observed = subset['Value'][subset['TimePeriod'].isin(X_test)]
    
        # calibration
        temp_popt = calibration(X_train, y_train)
        
        # logistic 
        simulated = logistic(X_test,temp_popt[0],temp_popt[1],temp_popt[2],temp_popt[3])
        simulated_values.append(simulated)
        
        #r2
        r2_value = r2_score(observed, simulated)
        r2_country.append(r2_value)
        
        #nrmse
        nrmse_value = nrmse(observed, simulated)
        nrmse_country.append(nrmse_value)
        
        #PBIAS
        pbias_value = pbias(observed, simulated)
        pbias_country.append(pbias_value)    
    
    r2_validation.append(np.array(r2_country).mean())
    nrmse_validation.append(np.array(nrmse_country).mean())
    pbias_validation.append(np.array(pbias_country).mean())
    
         
validated_data  = pd.DataFrame({'Country': unique_countries, 'r2_validation': r2_validation, 'nrmse_validation': nrmse_validation,'pbias_validation': pbias_validation })

#%%
# Select two contrasting countries and display their observed development 
# and simulated trend in one common time series plot 

#Colombia data
colombia_observed = df_final['Value'][df_final['GeoAreaName'] =='Colombia']







#%%
#Observed variables
colombia_observed = df_final['Value'][df_final['GeoAreaName'] =='Colombia']
# colombia_years = df_final['TimePeriod'][df_final['GeoAreaName']=='Colombia'].tolist()

kenya_observed = df_final['Value'][df_final['GeoAreaName']=='Kenya'].tolist()
# kenya_years = df_final['TimePeriod'][df_final['GeoAreaName']=='Kenya'].tolist()

#Variables Colombia

country='Colombia'
K_c= validated_data['K'].loc[validated_data['Country']==country].tolist()
r_c= validated_data['r'].loc[validated_data['Country']==country].tolist()
s_c=validated_data.start.loc[validated_data['Country']==country].tolist()
x_c=validated_data.x_peak.loc[validated_data['Country']==country].tolist()

#Variables Kenya
country='Kenya'
K_k= validated_data['K'].loc[validated_data['Country']==country].tolist()
r_k= validated_data['r'].loc[validated_data['Country']==country].tolist()
s_k=validated_data.start.loc[validated_data['Country']==country].tolist()
x_k=validated_data.x_peak.loc[validated_data['Country']==country].tolist()

#Timeperiod
t_pred=np.arange(2000,2030)
# t_obs=np.arange(2000,2021)

#Predicted variables
Colombia_pred_df=pd.DataFrame()
Colombia_pred_df['time']=t_pred
Colombia_pred_df['Predicted']=logistic(t_pred, s_c, K_c,x_c,r_c)
Colombia_pred = logistic(t_pred, s_c, K_c,x_c,r_c)
Kenya_pred=logistic(t_pred, s_k, K_k,x_k,r_k)

# %%Plotting the data of Colombia
fig= plt.figure(figsize = (16,9)) # figure size in 16:9 ratio
fig = sns.set(style="darkgrid")
# create scatter plot
# fig=sns.scatterplot(x = 'time', y='Predicted', data = Colombia_pred_df, facecolor="blue",
#                     edgecolor= 'blue', linewidth = 0.2)

fig=sns.scatterplot(x = t_pred, y=Colombia_pred, facecolor="red",edgecolor= 'red', linewidth = 3)
fig=sns.scatterplot(x = colombia_years, y = colombia_observed, facecolor="blue", edgecolor= 'blue', linewidth = 3)  

plt.xlabel("Time", fontsize = 20) # x-axis label
plt.xlim(2000, 2030)
plt.ylabel('Employed population below international poverty line, by sex and age (%)', fontsize = 16) # y-axis label
# plt.ylim(250,260)
plt.tick_params(labelsize = 14)
plt.legend(labels=["Simulated","Observed"],fontsize = 'xx-large')
# save generated scatter plot at program location
plt.savefig("Colombia.png") 
plt.show() # show scatter plot
plt.close()

#%%Plotting the data of KENYA
fig= plt.figure(figsize = (16,9)) # figure size in 16:9 ratio
fig= sns.set(style="darkgrid") 

# create scatter plot
fig=sns.scatterplot(x = t_pred, y=Kenya_pred, facecolor="red",edgecolor= 'red', linewidth = 3)
fig=sns.scatterplot(x = kenya_years, y = kenya_observed, facecolor="blue",edgecolor= 'blue', linewidth = 3) 

plt.xlabel("Time", fontsize = 20) # x-axis label
plt.xlim(2000, 2030)
plt.ylabel('Employed population below international poverty line, by sex and age (%)', fontsize = 16) # y-axis label
# plt.ylim(100,700)
plt.tick_params(labelsize = 14)
plt.legend(labels=["Simulated","Observed"],fontsize = 'xx-large')
plt.savefig("Kenya.png") # save generated scatter plot at program location
plt.close()





        


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