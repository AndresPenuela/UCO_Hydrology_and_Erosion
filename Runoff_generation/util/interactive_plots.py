#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

obs_data = pd.read_excel('data/data example 2.xlsx',index_col = 'date')

year_of_study = 2010
obs_data_year = obs_data[obs_data.index.year == year_of_study] # to select only data from to the year that we want to study

T = len(obs_data_year)
# Inputs
dates = obs_data_year.index
prec  = obs_data_year['rain']
etp   = obs_data_year['etp']
A = 500 * 10000 # ha to m2

def interactive_plot_1(p1=0.5, p2=0.5, p3=0.5, p4=0.5, p5=0.5): # valores iniciales de los parametros
    """
    This hydrological model is an adaptation of the HyMOD model. It has 5 parameters:

    soil_capacity: Maximum soil retention capacity (mm) [5-30]
    evap_ratio: Evapotranspiration ratio () [0-1]
    perc_ratio: Percolation ratio () [0-1]
    travel_time_surf: Concentration time of surface flow [0.8 - 2]
    travel_time_sub: Concentration time of subsurface flow [2 - 10]
    """
    
    # Assign blind variables and denormalize
    soil_capacity = p5 * (30 - 5) + 5
    evap_ratio = p3
    perc_ratio = p4
    travel_time_surf = p1 * (2 - 0.8) + 0.8
    travel_time_sub = p2 * (10 - 2) + 2
    
    #######################################################################
    # Initialization of variables
    #######################################################################
    effec_rain = np.zeros((T,1)) # Non-infiltrated rain [mm/t]
    et = np.zeros((T,1)) # Evapotranspiration rate [mm/t]
    sm = np.zeros((T+1,1)) # Soil moisture content [mm] (we assume that the soil is dry initially)
    sL = np.zeros((T+1,1)) # Slow reservoir moisture content [mm]
    sF = np.zeros((T+1,1)) # Fast reservoir moisture content [mm]
    q_sub = np.zeros((T,1)) # Underground flow [mm/t]
    q_sur = np.zeros((T,1)) # Overland flow [mm/t]

    #######################################################################
    # Simulation
    #######################################################################
    for t in range(1,T):

        ##### Effective rainfall (or rainfall excess) #####
        effec_rain[t] = max(sm[t-1] + prec[t] - soil_capacity, 0)

        ##### Temporary soil moisture #####
        sm_temp = min(max(sm[t-1] + prec[t], 0), soil_capacity)

        ##### Evapotranspiration #####
        W = min(np.abs(sm_temp/soil_capacity)*evap_ratio, 1) # Correction factor
        et[t] = W * etp[t] # Potential evapotranspiration (etp) to actual evapotranspiration (et)

        ##### Subsurface flow  or baseflow ######
        sL[t] = sL[t-1] + perc_ratio*effec_rain[t] - q_sub[t-1]
        q_sub[t] = 1/travel_time_sub * sL[t]

        ##### Surface flow or runoff #####
        sF[t] = sF[t-1] + (1-perc_ratio)*effec_rain[t] - q_sur[t-1]
        q_sur[t] = 1/travel_time_surf * sF[t]
        
        ##### Soil moisture at time t #####       
        sm[t] = min(max(sm[t-1] + prec[t] - et[t] - q_sub[t] - q_sur[t], 0), soil_capacity)

    Q_sub = q_sub * 0.001 * A # to convert mm/day into m3/day
    Q_sur = q_sur * 0.001 * A # to convert mm/day into m3/day
    
    ##### Total simulated flow #####
    Q_sim = Q_sur + Q_sub

    #######################################################################
    # Results visualization
    #######################################################################
    # Plot the figure
    plt.figure(figsize=(15,5))
    plt.plot(dates,Q_sim, label = 'total = surface + subsurface flow', color = 'blue', alpha = 0.5)
    #plt.plot(dates,Q_sur, linestyle = '--', color = 'orange',   label = 'surface flow or surface runoff')
    plt.plot(dates,Q_sub, linestyle = ':',  color = 'green', label = 'subsurface flow or baseflow')
    plt.ylim(0,300000) # graph axis limits
    plt.legend() # to show the legend
    plt.ylabel('m3/day') # label of the y axis
    plt.show()


