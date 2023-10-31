# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:01:41 2018

@author: ap18525
"""
import numpy as np

def hidrograma_historico(T,prec,etp,sm_max=40, ratio_evap=0.5, ratio_inf=0.75, t_sup=2, t_sub=8): # valores iniciales de los parametros
    """
    Este modelo hidrologico es una adaptacion del modelo HyMOD. Tiene 5 parametros:

    sm_max: Maxima capacidad de retencion del del suelo (mm) [10-90]
    ratio_evap:  Ratio de evapotranspiracion () [0-1]
    ratio_inf:  Ratio de infiltracion () [0-1]
    t_sup:  Tiempo concentracion del flujo superficial [0.8 - 2]
    t_sub:  Tiempo concentracion del flujo subterraneo [2 - 10]
    """
    
    #######################################################################
    # Inicializar variables
    #######################################################################
    inf   = np.zeros((T,1))   # Lluvia infiltrada [mm/t]
    sup   = np.zeros((T,1))   # Lluvia acumulada en superficie [mm/t]
    et    = np.zeros((T,1))   # Tasa de evapotranspiration [mm/t]
    sm    = np.zeros((T+1,1)) # Contendio de humdedad en el suelo [mm] (suponemos que el suelo esta seco inicialmente)
    sL    = np.zeros((T+1,1)) # Slow reservoir moisture [mm]
    sF    = np.zeros((T+1,1)) # Fast reservoir moisture [mm]
    Q_sub = np.zeros((T,1))   # Flujo subterraneo [mm/t]
    Q_sup = np.zeros((T,1))   # Flujo superficial [mm/t]
    Q_tot = np.zeros((T,1))   # Flujo total [mm/t]

    #######################################################################
    # Simulacion
    #######################################################################
    for t in range(T):

        ##### Humedad del suelo #####          
        sm_temp = max(min(sm[t] + prec[t], sm_max), 0)

        ##### Lluvia infiltrada #####
        inf[t] = max(sm[t] + prec[t] - sm_max, 0) * ratio_inf
        
        ##### Lluvia acumulada en superficie #####
        sup[t] = max(sm[t] + prec[t] - sm_max, 0) * (1- ratio_inf)

        ##### Evapotranspiracion #####
        W = min(np.abs(sm[t]/sm_max)*ratio_evap, 1) # Factor de correccion de la evapotranspiracion
        et[t] = W * etp[t] # Calculo de la evapotranspiracion

        ##### Humedad del suelo (t+1) #####
        sm[t+1] = max(min(sm_temp - et[t], sm_max), 0)

        ##### Flujo subterraneo ######
        Q_sub[t] = 1/t_sub * sL[t]
        sL[t+1] = sL[t] + inf[t] - Q_sub[t]

        ##### Flujo superficial #####
        Q_sup[t] = 1/t_sup * sF[t]
        sF[t+1] = sF[t] +  sup[t] - Q_sup[t]

        ##### Flujo total #####
        Q_tot[t] = Q_sup[t] + Q_sub[t]
        
    return Q_tot
