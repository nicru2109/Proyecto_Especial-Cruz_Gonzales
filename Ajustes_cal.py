##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Funciones as fn
import scipy.signal as sig
from f_SignalProcFuncLibs import *

##

Temp_read1 = pd.read_csv(os.path.join('initial_tests', 'p_derecha.txt'), header=4)
# Temp_read1 = pd.read_csv(os.path.join('initial_tests', 'OpenBCI-RAW-2022-05-10_20-37-00.txt'), header=4)
Temp1 = Temp_read1[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 3', ' EXG Channel 4',  ' EXG Channel 5', ' EXG Channel 6', ' EXG Channel 7']].to_numpy()

Tipo_Señal = 'EOG'
Tipo_Movimiento = 'Derecha'

Umbral = fn.Calibracion(Temp1, Tipo_Señal, Tipo_Movimiento)
