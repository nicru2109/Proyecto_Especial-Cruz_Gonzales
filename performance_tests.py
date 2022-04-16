# Script para las pruebas de rendimiento de la interfaz

## Imports
from Clases import *
from f_SignalProcFuncLibs import *
import numpy as np
import Funciones as fn
from pyOpenBCI import OpenBCICyton
import random

## Funciones
# Generador de movimientos aleatorios
def mov_list(mode='train', **movimientos):
    if mode=='train':
        reps = 3
    elif mode=='test':
        reps = 5
    else:
        raise Exception('Invalid mode. Expected "train" or "test"')

    movs = []

    for mov in movimientos:

        movs = movs + ([movimientos[mov]] * reps)

    random.shuffle(movs)

    return movs



# Pruebas EOG


