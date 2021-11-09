# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:46:32 2021

@author: patta
    
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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from assignment_B_model import logistic, calibration, nrmse, pbias, warn
import seaborn as sns
import warnings


# read data
df = pd.read_csv('Goal1.csv')

#%% Button

# 'Still working on this, it runs but not connected to code'

# import sys
# from PyQt5.QtWidgets import (QLabel, QRadioButton, QPushButton, QVBoxLayout, QApplication, QWidget)


# class basicRadiobuttonExample(QWidget):
    
#     global k_value 
    
#     def __init__(self):
#         super().__init__()

#         self.init_ui()

#     def init_ui(self):
#         self.label = QLabel('Select the cross-validation method')
#         self.rbtn1 = QRadioButton('5-fold cross validation')
#         self.rbtn2 = QRadioButton('10-fold cross validation')
#         self.label2 = QLabel("")
        
#         # self.rbtn1.toggled.connect(self.onClicked)
#         # self.rbtn2.toggled.connect(self.onClicked)

#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         layout.addWidget(self.rbtn1)
#         layout.addWidget(self.rbtn2)
#         layout.addWidget(self.label2)
        
#         self.setGeometry(600, 600,900, 50)
#         # self.setFont(QFont('Arial', 10)
#         self.setLayout(layout)
#         self.setWindowTitle('Input Needed')

#         # this will activate the window
#         self.activateWindow()        

#         self.show()
    
#     def dispK(self):
#         k_value=5
#         if self.rbtn1.isChecked()==True:
#             k_value=5
#         if self.rbtn1==True:
#             k_value=10
#         self.ui.labelFare.setText("Your selected k value is: "+str(k_value), 
#                                   ". Please allow a moment for the analysis to run.")
             
#     def onClicked(self):
#         radioBtn = self.sender()
#         if radioBtn.isChecked():
#             self.label2.setText("You live in " + radioBtn.text())


# if __name__ == '__main__':    
#     app = QApplication(sys.argv)
#     ex = basicRadiobuttonExample()
#     sys.exit(app.exec_())
#%% filter the data
# drop nan columns
df = df.drop(['Goal','Target','Source','Nature','Reporting Type','TimeCoverage','UpperBound', 'LowerBound','BasePeriod','GeoInfoUrl','FootNote'], axis=1)
df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]

# focusing on women between 15-24 only 
df_female = df.loc[df["Sex"]=="FEMALE"]
df_final = df_female.loc[df_female["Age"]=="15-24"]

#country list
unique_countries = pd.unique(df_final['GeoAreaName'].to_list())
#country code list
unique_codes = np.unique(df_final['GeoAreaCode']) 

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
# list to collect logistic values and calibration output
popt_list=[]
logistic_values=[]

# creating 2030 data
popt_2030 = []
logistic_2030 = []

for country in unique_countries:
    subset = df_final[df_final['GeoAreaName'] == country]
    year = subset['TimePeriod']
    value = subset['Value']
    temp_popt = calibration(year, value)
    popt_list.append(temp_popt)
    log_list = logistic(year,temp_popt[0],temp_popt[1],temp_popt[2],temp_popt[3])
    logistic_values.append(log_list)
    
    #log values for 2030
    log2030 = logistic(2030,temp_popt[0],temp_popt[1],temp_popt[2],temp_popt[3])
    logistic_2030.append(log2030)

#create a df with the 2030 logistic values
# validated_data  = pd.DataFrame ({'Country':unique_countries, 'Country_codes': unique_codes, '2030_projected_value':logistic_2030})   

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

# evaluation_df = pd.DataFrame({'Countries': unique_countries, 'r2': r2_values, 'nrmse': nrmse_values, 'pbias': pbias_values})

#%% Validate using 5-fold cross validation
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

#Storing          
# validated_data  = pd.DataFrame({'Country': unique_countries, 'r2_validation': r2_validation, 'nrmse_validation': nrmse_validation,'pbias_validation': pbias_validation })
calibration_data = pd.DataFrame({'Country': unique_countries, 'country_code': unique_codes, 'Calib_info': popt_list})
split_df = pd.DataFrame(calibration_data['Calib_info'].to_list(), columns = ['x_start', 'K', 'x_peak', 'r'])
calibration_df = pd.concat([calibration_data, split_df], axis=1)
calibration_df = calibration_df.drop('Calib_info', axis=1)

#%%creating a csv to export all data
indicator111_df = pd.DataFrame({'Country': unique_countries, 'Growth_rate': calibration_df['r'], 'Values_2030': logistic_2030, 'r2_evaluation': r2_values, 'nrmse_evaluation': nrmse_values, 'pbias_evaluation': pbias_values, 'r2_validation': r2_validation, 'nrmse_validation': nrmse_validation,'pbias_validation': pbias_validation})
indicator111_df.insert(0,'Indicator', '1.1.1') # write indicator
indicator111_df.insert(1,'Description', 'Employed population below international poverty line, by sex and age (%)') # write indicator description

indicator111_df.to_csv(r'Indicator111_information.csv')

#%%
# Select two contrasting countries and display their observed development 
# and simulated trend in one common time series plot 

#Colombia data
country = 'Colombia'

colombia_observed = df_final['Value'][df_final['GeoAreaName'] == country]
colombia_value = calibration_df[calibration_df['Country'] == country]
year_colombia = df_final['TimePeriod'][df_final['GeoAreaName'] == country]
# year_colombia = np.arange(2000,2030)
colombia_simulated = logistic(year_colombia, colombia_value['x_start'].to_list(),colombia_value['K'].to_list(), colombia_value['x_peak'].to_list(),colombia_value['r'].to_list())

#Kenya data
country = 'Kenya'
kenya_observed = df_final['Value'][df_final['GeoAreaName'] == country]
kenya_value = calibration_df[calibration_df['Country'] == country]
year_kenya = df_final['TimePeriod'][df_final['GeoAreaName'] == country]
# year_kenya = np.arange(2000,2030)
kenya_simulated = logistic(year_kenya, kenya_value['x_start'].to_list(),kenya_value['K'].to_list(), kenya_value['x_peak'].to_list(),kenya_value['r'].to_list())

#%% plotting data
fig= plt.figure(figsize = (20,12)) #creating fig and fig size
fig = sns.set_style("ticks")

# plot colombia data
fig=sns.scatterplot(x = year_colombia, y = colombia_observed, facecolor="blue", edgecolor= 'blue', linewidth = 5)  
fig=sns.lineplot(x = year_colombia, y=colombia_simulated, palette='deepskyblue', linewidth = 5)

# plot kenya data
fig=sns.scatterplot(x = year_kenya, y = kenya_observed, facecolor="orangered", edgecolor= 'orangered', linewidth = 5)  
fig=sns.lineplot(x = year_kenya, y=kenya_simulated, palette='red', linewidth = 5)

plt.xlabel("Time", fontsize = 20) # x-axis label
plt.xticks(np.arange(2000,2019,1)) # formatting the x-axis
plt.tick_params(labelsize = 18) # size of the axis
plt.ylabel('Employed population below international poverty line, by sex and age (%)', fontsize = 20) # y-axis label
plt.ylim(bottom = 0)
plt.legend(labels=["Simulated Colombia","Simulated Kenya","Observed Colombia","Observed Kenya"],fontsize = 'xx-large')  # set the legend

plt.savefig("scatterplot_%_women_15to24_below_international_poverty_line_2000-2018_Colombia_Kenya.png", bbox_inches = 'tight',dpi =300) # save plot
plt.close()

#%% Task 5 - Plotting most populous 30 countries

# UN population data 
women_population = pd.read_excel('WPP2019_POP_F01_3_TOTAL_POPULATION_FEMALE.xlsx', skiprows=16,usecols= ["Region, subregion, country or area *", "Country code","Type", "2020"])
women_population=women_population[women_population['Type'] == 'Country/Area']   #filtering data to collect countries

# order UN data based on descending population
descending_values = women_population.sort_values('2020', ascending=False)
descending_values['country_code'] = descending_values["Country code"]   # prepare data to drop information
descending_values = descending_values.drop(columns=["Country code"])

joined_df = descending_values.set_index('country_code').join(calibration_df.set_index('country_code'))  #join the df togther
top30 = joined_df.dropna()  #drop countries which are not in SDG data
top30 = top30[:30]  #take the top 30
top30 = top30.sort_values('r') # order the values by lowest to highest growth rate 


#plot
fig= plt.figure(figsize = (20,12)) #creating fig and fig size
fig, ax = plt.subplots(1)  
fig = sns.set_style("ticks")

plt.scatter(x = top30['r'], 
                          y = top30['Country'], 
                          c = top30['r'],
                          cmap= 'jet')

ax.set_xlabel('Growth rate', fontsize = 12)
ax.set_ylabel('Countries', fontsize = 12)

plt.grid(alpha=0.3)

plt.colorbar().set_label('Projected Indicator Growth', fontsize = 12)
plt.setp(ax.get_yticklabels(), fontsize='7.5') 

plt.savefig("TOP30_scatterplot_%_women_15to24_below_international_poverty_line_growth_rate.png", bbox_inches = 'tight',dpi = 300) # save plot
plt.close()     


    

