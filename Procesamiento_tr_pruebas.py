##

from Clases import *
from f_SignalProcFuncLibs import *
import numpy as np
import Funciones as fn
from pyOpenBCI import OpenBCICyton
import random
import scipy.signal as sig
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt

## Constantes

s_SRate = 250 # Hertz
window = 1 # segundos
act = 0.25 # segundos
tspan_exps = 25 * s_SRate

# Configuración de la board
board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count

# Filtros
filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

# Lista inicial
inc_data = []

# Lista procesada
pr_data = []

# Cargar umbrales previamente calculados en calibracion
U_Parpadeo = np.array(np.loadtxt('trhlds//U_Parpadeo.txt'))
U_Derecha_EOG = np.array(np.loadtxt('trhlds//U_Derecha_EOG.txt'))
U_Izquierda_EOG = np.array(np.loadtxt('trhlds//U_Izquierda_EOG.txt'))

U_Arriba = np.array(np.loadtxt('trhlds//U_Arriba.txt'))
U_Arriba = U_Arriba[0]

U_Izquierda_EMG = np.array(np.loadtxt('trhlds//U_Izquierda_EMG.txt'))
U_Izquierda_EMG = U_Izquierda_EMG[0]

U_Derecha_EMG = np.array(np.loadtxt('trhlds//U_Derecha_EMG.txt'))
U_Derecha_EMG = U_Derecha_EMG[0]

Movimiento = pre_wind()
ventana = proc_wind(8, int(s_SRate * window), int(act * s_SRate))

def adquisicion(sample):
    if len(inc_data) == 0:
        pygame.mixer.init()
        pygame.mixer.music.load("mp3//go.mp3")
        pygame.mixer.music.play()

    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    if (len(inc_data) % int(s_SRate * act)) == 0:
        ventana.refresh(inc_data[-int(s_SRate * act):])
        procesamiento(ventana.data)
    if len(inc_data) == tspan_exps:
        board.stop_stream()


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

    mov = fn.identificar_movimiento(diff_der_eog_avg, diff_izq_eog_avg, U_Derecha_EOG, U_Izquierda_EOG, U_Parpadeo)

    print(mov)

    pr_data.append(np.array([sig_arr_emg[-int(s_SRate * act):], sig_der_emg[-int(s_SRate * act):], sig_izq_emg[-int(s_SRate * act):], diff_der_eog_avg[-int(s_SRate * act):], diff_izq_eog_avg[-int(s_SRate * act):]]))


##

rec = input('Desea tomar datos? (si/no) ')
nombre = input('Indique el nombre del archivo ')

if rec == 'si':
    board.start_stream(adquisicion)

    l = len(pr_data)
    z = pr_data[0].shape[1]

    data = np.zeros((pr_data[0].shape[0], l * z))

    for i in range(l):
        data[:, (i * z):((i + 1) * z)] = pr_data[i]

    np.savetxt('initial_tests//' + nombre + '.txt', data)

##

data_pr = np.array(np.loadtxt('initial_tests//' + nombre + '.txt'))

## Visualización inicial

plt.figure()
plt.plot(data_pr[3, :], 'b')
plt.plot(data_pr[4, :], 'r')
# plt.axline(xy1=[0, U_Parpadeo[0]], slope=0, color='b')
# plt.axline(xy1=[0, U_Parpadeo[1]], slope=0, color='r')
# plt.axline(xy1=[0, U_Parpadeo[2]], slope=0, color='b')
# plt.axline(xy1=[0, U_Parpadeo[3]], slope=0, color='r')
# plt.axline(xy1=[0, U_Derecha_EOG[0]], slope=0, color='b')
# plt.axline(xy1=[0, U_Derecha_EOG[1]], slope=0, color='r')
# plt.axline(xy1=[0, U_Derecha_EOG[2]], slope=0, color='b')
# plt.axline(xy1=[0, U_Derecha_EOG[3]], slope=0, color='r')
plt.axline(xy1=[0, U_Izquierda_EOG[0]], slope=0, color='b')
plt.axline(xy1=[0, U_Izquierda_EOG[1]], slope=0, color='r')
plt.axline(xy1=[0, U_Izquierda_EOG[2]], slope=0, color='b')
plt.axline(xy1=[0, U_Izquierda_EOG[3]], slope=0, color='r')
plt.title('Señal procesada')
plt.xlabel('No. Muestra')
plt.ylabel('Amplitud [mV]')

