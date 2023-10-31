# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:01:41 2018

@author: ap18525
"""
import numpy as np

def onda_ref():
    amp = 2.7 # amplitud
    fase = 0.6 # fase
    frec = 4.2 # frecuencia
    x = np.arange(0,1,1/500)
    y = amp * np.sin(2 * np.pi * (frec * x + fase))
    return x,y
