# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:01:41 2018

@author: ap18525
"""
import numpy as np

def sound_wave_ref():
    amp = 3 # amplitude
    phase = 0.6 # phase
    freq = 4 # frequency
    x = np.linspace(0,1,500) # x axis from 0 to 1 with a 1/500 step
    y = amp * np.sin(2 * np.pi * (freq * x + phase))
    return x,y

def sound_wave_blind(p1,p2,p3):
    x = np.linspace(0,1,500) # x axis from 0 to 1 with a 1/500 step
    y = p2*20 * np.sin(2 * np.pi * (p3*10 * x + p1))
    return x,y
