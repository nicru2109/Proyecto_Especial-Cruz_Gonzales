##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Funciones as fn
import scipy.signal as sig
from f_SignalProcFuncLibs import *
import pygame
from Clases import *
from pyOpenBCI import OpenBCICyton

## Constantes

s_SRate = 250 # Hertz
window = 1 # segundos
act = 0.25 # segundos

# Configuración de la board

board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count

# Filtros

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

# Lista inicial
inc_data = []

# Umbrales
U_Parpadeo = np.array(np.loadtxt('trhlds//U_Parpadeo.txt'))
U_Derecha_EOG = np.array(np.loadtxt('trhlds//U_Derecha_EOG.txt'))
U_Izquierda_EOG = np.array(np.loadtxt('trhlds//U_Izquierda_EOG.txt'))

U_Arriba = np.array(np.loadtxt('trhlds//U_Arriba.txt'))
U_Arriba = U_Arriba[0]

U_Izquierda_EMG = np.array(np.loadtxt('trhlds//U_Izquierda_EMG.txt'))
U_Izquierda_EMG = U_Izquierda_EMG[0]

U_Derecha_EMG = np.array(np.loadtxt('trhlds//U_Derecha_EMG.txt'))
U_Derecha_EMG = U_Derecha_EMG[0]

# Objetos

Movimiento = pre_wind()
ventana = proc_wind(8, int(s_SRate * window), int(act * s_SRate))

## Funciones
# Adquisición

def adquisicion(sample):
    if len(inc_data) == 0:
        pygame.mixer.init()
        pygame.mixer.music.load("mp3//go.mp3")
        pygame.mixer.music.play()

    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    if (len(inc_data) % int(s_SRate * act)) == 0:
        ventana.refresh(inc_data[-int(s_SRate * act):])
        procesamiento(ventana.data)

# Procesamiento

def procesamiento(data):
    sig_arr_emg = sig.detrend(data[:, 0])
    sig_der_emg = sig.detrend(data[:, 1])
    sig_izq_emg = sig.detrend(data[:, 2])
    sig_der_eog = sig.detrend(data[:, 4])
    sig_izq_eog = sig.detrend(data[:, 5])

    # Filtro

    sig_izq_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_izq_emg)
    sig_der_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_der_emg)
    sig_arr_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_arr_emg)
    sig_izq_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_izq_eog)
    sig_der_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_der_eog)

    # Artefactos en los bordes
    sig_arr_emg = sig_arr_emg[int(0.1 * window * s_SRate):-int(0.1 * window * s_SRate)]
    sig_der_emg = sig_der_emg[int(0.1 * window * s_SRate):-int(0.1 * window * s_SRate)]
    sig_izq_emg = sig_izq_emg[int(0.1 * window * s_SRate):-int(0.1 * window * s_SRate)]
    sig_der_eog = sig_der_eog[int(0.1 * window * s_SRate):-int(0.1 * window * s_SRate)]
    sig_izq_eog = sig_izq_eog[int(0.1 * window * s_SRate):-int(0.1 * window * s_SRate)]

    # Suavizado
    sig_izq_eog_avg = fn.f_AvFlt(sig_izq_eog, s_SRate, 0.2)
    sig_der_eog_avg = fn.f_AvFlt(sig_der_eog, s_SRate, 0.2)

    # Primera derivada

    diff_izq_eog = np.diff(sig_izq_eog_avg)
    diff_der_eog = np.diff(sig_der_eog_avg)

    # Suavizado 2

    diff_izq_eog_avg = fn.f_AvFlt(diff_izq_eog, s_SRate, 0.2)
    diff_der_eog_avg = fn.f_AvFlt(diff_der_eog, s_SRate, 0.2)

    # Maximo de ventana

    diff_izq_emg_avg = np.max(sig_izq_emg)
    diff_der_emg_avg = np.max(sig_der_emg)
    diff_arr_emg_avg = np.max(sig_arr_emg)

    # Movimiento EOG

    mov = fn.identificar_movimiento(diff_der_eog_avg, diff_izq_eog_avg, U_Derecha_EOG, U_Izquierda_EOG, U_Parpadeo)
    Movimiento.actualizar(mov)

    # Movimiento EMG

    if diff_arr_emg_avg > U_Arriba and not diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG:
        mov_emg = 'MF'
        print(mov_emg)
        Movimiento.actualizar(mov)

    elif diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG and not diff_arr_emg_avg > U_Arriba:
        mov_emg = 'CD'
        print(mov_emg)
        Movimiento.actualizar(mov)

    elif diff_izq_emg_avg > U_Izquierda_EMG and not diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
        mov_emg = 'CI'
        print(mov_emg)
        Movimiento.actualizar(mov)

    elif diff_izq_emg_avg > U_Izquierda_EMG and diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
        mov_emg = 'C'
        print(mov_emg)
        Movimiento.actualizar(mov)

    else:
        mov_emg = 'Nada'
        Movimiento.actualizar(mov)

## Iniciar toma de datos

board.start_stream(adquisicion)
