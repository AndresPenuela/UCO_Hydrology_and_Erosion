#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from scipy.optimize import newton, minimize
from math import log
import numpy as np
import matplotlib.pyplot as plt

# Green-Ampt model
def G(K_s, Ψ, Δθ, F, t, F_p, t_p): 
    """
    Function G to calculate the cumulative infiltration F for a given time step t.
    The solution of the function, G(F) = 0, must be done by trial-and-error or by 
    some numerical method, such as Newton-Raphson
    """
    return K_s * (t - t_p) - F + F_p + Δθ * Ψ * np.log((F + Δθ * Ψ)/(F_p + Δθ * Ψ))

r = 40 # rainfall rate (mm/h)

def GA_interactive_1(soil_type = 'loamy sand'):
    
    T = 24 # hours 
    
    if soil_type == 'loamy sand':
        K_s = 29.9; Ψ = 61.3;  η = 0.437; θ_e = 0.417
    elif soil_type == 'sandy loam':
        K_s = 10.9; Ψ = 110.1;  η = 0.453; θ_e = 0.412
    elif soil_type == 'loam':
        K_s = 3.4;  Ψ = 88.9;  η = 0.463; θ_e = 0.434
    elif soil_type == 'clay loam':
        K_s = 1;  Ψ = 270.3;  η = 0.471; θ_e = 0.432
    elif soil_type == 'clay':
        K_s = 0.3;  Ψ = 316.3; η = 0.475; θ_e = 0.385
    
    θ_i = 0.15
    Δθ = η - θ_i
        
    F_all = np.zeros(T+2)
    t_all = np.zeros(T+2)
    f_all = np.zeros(T+2)
    e_all = np.zeros(T+2)
    E_all = np.zeros(T+2)
    
    #The ponding time `tp` under constant rainfall intensity using the Green-Ampt infiltration equation
    t_p = K_s * Ψ * Δθ / (r * (r - K_s))
    i = int(np.ceil(t_p))
    t_all[i] = t_p
    #Then `F` during the ponding time is:
    F_p = r * t_p # f = i during the ponding time
    F_all[i] = F_p
    f_all[0:i] = r 
    f_all[i] = F_all[i] / t_all[i]
    e_all[i] = r - f_all[i]
    E_all[i] = e_all[i] * t_all[i]

    for t in range(i+1,T+2):

        t_all[t] = t
        fun = lambda F: G(K_s, Ψ, Δθ, F, t, F_p, t_p)
        F_all[t] = newton(fun,x0 = 3)

        f_all[t] = min(r,(F_all[t] - F_all[t-1]) / (t_all[t] - t_all[t-1]))
        e_all[t] = r - f_all[t]
        E_all[t] = E_all[t-1] + e_all[t] * (t_all[t] - t_all[t-1])

    L = F_all/Δθ
    
    # Plot wetting front diagram
    plt.figure(figsize=(15,4)) # to define the plot size
    plt.subplot(1,3,1)
    plt.hlines(0,0,1, color = 'black')
    for t in range(i,T+2):
        plt.hlines(-L[t],θ_i,η, color = 'blue',alpha=t/(T+1))
#    plt.hlines(-L[i],θ_i,η, color = 'grey', linestyles='--',label = "wetting front at tp")
    plt.hlines(-L[-1],θ_i,η, color = 'blue', label = "wetting front")
    plt.vlines(η, -L[-1], 0, color = 'blue')
    plt.vlines(θ_i, -10000, 0, color = 'blue', linestyles=':', label = 'initial soil moisture')
    plt.vlines(θ_i, -10000, -L[-1], color = 'blue')
    plt.xlim([0,0.6])
    plt.ylim([-L[-1]*1.1,1])
    #plt.yscale('symlog')
    plt.xlabel('soil water content θ')
    plt.ylabel('soil depth (mm)')
    plt.title("Green-Ampt wetting front")
    plt.legend()
    
    # Plot cumulative infiltration
    plt.subplot(1,3,2)
    plt.plot(t_all,F_all, color = 'red', label = 'cumulative infiltration')
    plt.plot(t_all,E_all, color = 'blue', label = 'cumulative runoff')
#    plt.vlines(t_p, 0, 1000, linestyles='--', color = 'gray', label = 'ponding time = %.3f h' % t_p)
    plt.xlim([0,T])
    plt.ylim([0,F_all[-1]*1.1])
    plt.xlabel('hours')
    plt.ylabel('mm')
    plt.ylim([0,1000])
    plt.title('Cumulative infiltration & Runoff')
    plt.legend(loc = 'upper right')
    
    # Plot instant infiltration vs runoff
    plt.subplot(1,3,3)
    plt.hlines(r,0,T, color = 'lightblue', label = 'rainfall rate')
    plt.plot(t_all,f_all, color = 'red', label = 'infiltration rate')
    plt.plot(t_all,e_all, color = 'blue', label = 'runoff rate')
#    plt.vlines(t_p, 0, r*1.1, linestyles='--', color = 'gray', label = 'ponding time = %.3f h' % t_p)
    plt.xlim([0,T])
    plt.ylim([0,r*1.1])
    plt.xlabel('hours')
    plt.ylabel('mm/h')
    plt.title('Instant infiltration & Runoff')
    plt.legend()

def GA_interactive_2(p1 = 0.5, p2 = 60, p3 = 0.38):
    
    T = 24 # hours
        
    F_all = np.zeros(T+2)
    t_all = np.zeros(T+2)
    f_all = np.zeros(T+2)
    e_all = np.zeros(T+2)
    E_all = np.zeros(T+2)
    
    θ_i = 0.15
    Δθ = p3 - θ_i
    
    if p1 >= r:
        p1 = r-0.01
        raise ValueError('p1 is too high!')
        
    F_all = np.zeros(T+2)
    t_all = np.zeros(T+2)
    f_all = np.zeros(T+2)
    e_all = np.zeros(T+2)
    E_all = np.zeros(T+2)
    
    #The ponding time `tp` under constant rainfall intensity using the Green-Ampt infiltration equation
    t_p = p1 * p2 * Δθ / (r * (r - p1))
    i = int(np.ceil(t_p))
    
    if t_p > T:
        t_p = T
        raise ValueError('error: ponding time is too long!')
    
    t_all[i] = t_p
    #Then `F` during the ponding time is:
    F_p = r * t_p # f = i during the ponding time
    F_all[i] = F_p
    f_all[0:i] = r 
    f_all[i] = F_all[i] / t_all[i]
    e_all[i] = r - f_all[i]
    E_all[i] = e_all[i] * t_all[i]

    for t in range(i+1,T+2):

        t = t
        t_all[t] = t
        fun = lambda F: G(p1, p2, Δθ, F, t, F_p, t_p)
        F_all[t] = newton(fun,x0 = 3)

        f_all[t] = min(r,(F_all[t] - F_all[t-1]) / (t_all[t] - t_all[t-1]))
        e_all[t] = r - f_all[t]
        E_all[t] = E_all[t-1] + e_all[t] * (t_all[t] - t_all[t-1])

    L = F_all/Δθ
    
    # Plot wetting front diagram
    plt.figure(figsize=(15,4)) # to define the plot size
    plt.subplot(1,3,1)
    for t in range(i,T+2):
        plt.hlines(-L[t],θ_i,p3, color = 'blue',alpha=t/(T+1))
#    plt.hlines(-L[i],θ_i,p3, color = 'grey', linestyles='--',label = "wetting front at tp")
    plt.hlines(-L[-1],θ_i,p3, color = 'blue', label = "wetting front")
    plt.vlines(p3, -L[-1], 0, color = 'blue')
    plt.vlines(θ_i, -10000, 0, color = 'grey', linestyles=':', label = 'initial soil moisture')
    plt.vlines(θ_i, -10000, -L[-1], color = 'blue')
    plt.xlim([0,0.6])
    plt.ylim([-L[-1]*1.1,1])
    plt.xlabel('soil water content θ')
    plt.ylabel('soil depth (mm)')
    plt.title("Green-Ampt wetting front")
    plt.legend()
    
    # Plot cumulative infiltration
    plt.subplot(1,3,2)
    plt.plot(t_all,F_all, color = 'red', label = 'cumulative infiltration')
    plt.plot(t_all,E_all, color = 'blue', label = 'cumulative runoff')
#    plt.vlines(t_p, 0, 1000, linestyles=':', color = 'gray', label = 'ponding time tp = %.3f h' % t_p)
    plt.xlim([0,T])
    plt.ylim([0,1000])
    plt.xlabel('hours')
    plt.ylabel('mm')
    plt.title('Cumulative infiltration & Runoff')
    plt.legend()
    
    # Plot instant infiltration vs runoff
    plt.subplot(1,3,3)
    plt.hlines(r,0,T, color = 'lightblue', label = 'rainfall rate')
    plt.plot(t_all,f_all, color = 'red', label = 'infiltration rate')
    plt.plot(t_all,e_all, color = 'blue', label = 'runoff rate')
#    plt.vlines(t_p, 0, r*1.1, linestyles=':', color = 'gray', label = 'ponding time tp = %.3f h' % t_p)
    plt.xlim([0,T])
    plt.ylim([0,r*1.1])
    plt.xlabel('hours')
    plt.ylabel('mm/h')
    plt.title('Instant infiltration & Runoff')
    plt.legend()