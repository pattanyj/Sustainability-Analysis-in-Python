# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:46:32 2021

@author: patta
"""
# imports 
import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# # check path
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
countries = df_final['GeoAreaName'].to_list()
unique_countries = pd.unique(countries)

#%% sanity checks for each country
for country in unique_countries:
    subset = df_final[df_final['GeoAreaName'] == country]
    minimum=subset['Value'].min()
    maximum=subset['Value'].max()
    avg=subset['Value'].mean()
    print(country," : ", 
          'The minimum value is', minimum,
          'The maximum value is', maximum,
          'The average value is', avg)
    
#%% main analysis
def logistic(x, start, K, x_peak, r):
    """
    Logistic model
    
    This function runs a logistic model.
    
    Args:
        x (array_like): The control variable as a sequence of numeric values \
        in a list or a numpy array.
        start (float): The initial value of the return variable.
        K (float): The carrying capacity.
        x_peak (float): The x-value with the steepest growth.
        r (float): The growth rate.
        
    Returns:
        array_like: A numpy array or a single floating-point number with \
        the return variable.
    """
    
    if isinstance(x, list):
        x = np.array(x)
    return start + K / (1 + np.exp(r * (x_peak-x)))

def calibration(x, y):
    """
    Calibration
    
    This function calibrates a logistic model.
    The logistic model can have a positive or negative growth.
    
    Args:
        x (array_like): The explanatory variable as a sequence of numeric values \
        in a list or a numpy array.
        y (array_like): The response variable as a sequence of numeric values \
        in a list or a numpy array.
        
    Returns:
        tuple: A tuple including four values: 1) the initial value (start), \
        2) the carrying capacity (K), 3) the x-value with the steepest growth \
        (x_peak), and 4) the growth rate (r).
    """
    if isinstance(x, pd.Series): x = x.to_numpy(dtype='int')
    if isinstance(y, pd.Series): y = y.to_numpy(dtype='float')
    
    if len(np.unique(y)) == 1:
        return y[0], 0, 2000.0, 0
    
    # initial parameter guesses
    slope = [None] * (len(x) - 1)
    for i in range(len(slope)):
        slope[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        slope[i] = abs(slope[i])
    x_peak = x[slope.index(max(slope))] + 0.5
    
    if y[0] < y[-1]: # positive growth
        start = min(y)
        K = 2 * (sum([y[slope.index(max(slope))], \
                        y[slope.index(max(slope))+1]])/2 - start)
    else: # negative growth
        K = 2 * (max(y) - sum([y[slope.index(max(slope))], \
                        y[slope.index(max(slope))+1]])/2)
        start = max(y) - K
    
    if start <0:
        start = 0
        
    # curve fitting
    popt, _ = curve_fit(logistic, x, y, p0 = [start, K, x_peak, 0], maxfev = 10000,
                        bounds = ([0.5*start, 0.5*K, 1995, -10],
                                  [2*(start+0.001), 2*K, 2030, 10]))
    # +0.001 so that upper bound always larger than lower bound even if start=0
    return popt

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
    
    
#%% Evaluate the model using R2
from sklearn.metrics import r2_score, mean_squared_error

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

#%%PBIAS
#create new df to seperate by country
data ={'Countries':countries, 'Years':df_final['TimePeriod'],  'Y_true':y_true,'Y_pred':y_pred}
pbias_df = pd.DataFrame(data)
# pbias_df=pbias_df.groupby(Countries)      #this doesnt work

pbias_values=[]
for country in unique_countries:
    subset = pbias_df[pbias_df['Countries'] == country]
    y_true_1 = subset['Y_true']
    y_pred_1 = subset['Y_pred']
    y_true_1= np.array(y_true_1)
    y_pred_1=np.array(y_pred_1)
    pbias = 100* ((sum(y_pred_1-y_true_1))/sum(y_pred_1))
    pbias_values.append(pbias)

#%% 
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

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