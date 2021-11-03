import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

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

def nrmse(observed_values,simulated_values):
    """
    Normal Root Mean Squared (nmrse)
    
    This function calculated the normal root mean squared value.
    The nmrse model is a measure of the mean relative scatter and reflects the random errors.
    
    Args:
        observed_values (array_like): The actual values which were collected from the data \
        in a list or a numpy array.
        simulated_values (array_like): The simulated values which were collected from the logistic function \
        in a list or a numpy array.
        
    Returns:
        float:  a float value containing the nrmse.
    """
    rmse=mean_squared_error(observed_values, simulated_values)
    avg_y_true = sum(observed_values)/len(observed_values)
    
    if avg_y_true == 0:
        nrmse = np.nan
    else:
        nrmse = rmse/avg_y_true
    
    return nrmse

def pbias(observed_values,simulated_values):
    """
    Percent bias (pbias)
    
    This function calculated the percent bias.
    The pbias measures the average tendency of the simulated values to be larger or 
    smaller than their observed ones.
    
    Args:
        observed_values (array_like): The actual values which were collected from the data \
        in a list or a numpy array.
        simulated_values (array_like): The simulated values which were collected from the logistic function \
        in a list or a numpy array.
        
    Returns:
        float:  a float value containing the pbias.
    """
    
    if sum(np.array(simulated_values)) == 0:
        pbias = np.nan
    else:
        pbias = 100 * ((sum(np.array(simulated_values)-np.array(observed_values)))/sum(np.array(simulated_values)))

    return pbias

def warn(*args, **kwargs):
    pass





    
    
    