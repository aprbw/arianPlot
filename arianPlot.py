"""
arianPlot.py

This module provides a collection of function to visualize time-series data. 

-------------
- numpy (as np)
- matplotlib.pyplot (as plt)

by Arian Prabowo https://github.com/aprbw

"""

import numpy as np
import matplotlib.pyplot as plt

def heatmap(data,
            temporal_bin_size=None,
            is_sort=False,
            is_norm=True,
            red_aggegation_function = np.mean,
            green_aggegation_function = np.median,
            blue_aggegation_function = np.min,
            is_plot=True):
    """
    Plot heatmap of time-series data.

    Args:
        data: input data, shape (n_timestep, n_channel)
        temporal_bin_size (int): temporal bin size for heatmap
        sort (bool): sort the heatmap based on the mean of each channel
        is_norm (bool): normalize the heatmap per channel
        red_aggegation_function: red channel aggregation function
        green_aggegation_function: green channel aggregation function
        blue_aggegation_function: blue channel aggregation function
        is_plot (bool): call plt.imshow() to plot the heatmap

    Return:
        ahm: array of heatmap, shape (n_timestep, n_channel, 3)

    Note:
        The shape between input data and output ahm is transposed.
            This is because, usually, the input is a csv file where each row is a timestep.
            But x-axis of the heatmap is the time axis.
        Aggregation function options must take an array and axis of aggregation as input.
            The shape of array is (n_timestep, n_channel)
            The axis of aggregation is 0
            The output of aggregation function is a 1-D array of shape (n_channel)
            Example: np.mean, np.std, np.median, np.min, np.max
    """
    if temporal_bin_size is None:
        temporal_bin_size = data.shape[0] // (3 * data.shape[1])
    # ahm = array of heat map
    ahm = np.zeros((data.shape[1], data.shape[0]//temporal_bin_size, 3))
    for i in range(ahm.shape[1]):
        ahm[:,i,0] = red_aggegation_function(data[i*temporal_bin_size:(i+1)*temporal_bin_size,:], 0)
        ahm[:,i,1] = green_aggegation_function(data[i*temporal_bin_size:(i+1)*temporal_bin_size,:], 0)
        ahm[:,i,2] = blue_aggegation_function(data[i*temporal_bin_size:(i+1)*temporal_bin_size,:], 0)
    # min max noramlize ahm [0,1] per channel
    if is_norm:
        ahm -= np.nanmin(ahm, 1)[:,None,:]
        ahm /= np.nanmax(ahm, 1)[:,None,:]
    # sort
    if is_sort:
        ai_sort = np.argsort(np.nanmean(data, 1))
        ahm = ahm[ai_sort,:,:]
    # plot
    if is_plot:
        ahm = np.nan_to_num(ahm)
        plt.imshow(ahm, aspect='auto', interpolation='none')
    return ahm

def timeOfPeriod(l_data, n_period, labels=[], alpha_row=.05, alpha_mean=.6):
    """
    Plot time-of-period of time-series data. E.g. time-of-day or day-of-week.

    Args:
        l_data: list of input data, shape (n_timestep, n_channel)
        n_period: number of period in timestep
        labels: list of label for data array in l_data
        alpha_row: transparency of row
        alpha_mean: transparency of mean
    
    """
    if labels:
        assert len(labels) == len(l_data)
    # initialize latop = list of array of time of period
    latop = []
    for idata in l_data:
        # atop = array of time of period
        atop = np.zeros((n_period, idata.shape[1]))
        latop.append(atop)
    for i in range(n_period):
        for j in range(len(l_data)):
            idata = l_data[j]
            atop = latop[j]
            atop[i,:] = idata[i::n_period,:].mean(0)
            latop[j] = atop
    for i in range(len(latop)):
        plt.plot(latop[i], alpha=alpha_row, color='C'+str(i))
        plt.plot(latop[i].mean(1),
                 alpha=alpha_mean,
                 linewidth=3,
                 color='C'+str(i),
                 label=labels[i] if labels else ''
                 )
    return latop

def symlog(data):
    # symmetric log for scaling data for data viz
    data = data.copy()
    data[data>10] = np.log10(data[data>10])
    data[data<-10] = -np.log10(-data[data<-10])
    return data
