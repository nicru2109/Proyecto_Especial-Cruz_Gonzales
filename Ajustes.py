##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Funciones as fn
import scipy.signal as sig
from f_SignalProcFuncLibs import *

##

#Constantes
s_SRate = 250  # Hertz
window = 1 # segundos
act = 0.25 # segundos

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 50], [15, 52])

##

# Temp_read1 = pd.read_csv(os.path.join('initial_tests', 'prueba.txt'), header=4)
Temp_read1 = pd.read_csv(os.path.join('initial_tests', 'prueba2.txt'), header=4)
Temp1 = Temp_read1[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 4',  ' EXG Channel 5']].to_numpy()
Temp1 = Temp1[250:, :]

# v_1 = np.squeeze(Temp1[:, 4])

def procesamiento(data):
    sig_arr_emg = sig.detrend(data[:, 0])
    sig_der_emg = sig.detrend(data[:, 1])
    sig_izq_emg = sig.detrend(data[:, 2])
    sig_der_eog = sig.detrend(data[:, 3])
    sig_izq_eog = sig.detrend(data[:, 4])

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
    sig_izq_eog_avg = fn.f_AvFlt(sig_izq_eog, s_SRate, 0.08)
    sig_der_eog_avg = fn.f_AvFlt(sig_der_eog, s_SRate, 0.08)

    # Primera derivada

    diff_izq_eog = np.diff(sig_izq_eog_avg)
    diff_der_eog = np.diff(sig_der_eog_avg)

    # Suavizado 2

    diff_izq_eog_avg = fn.f_AvFlt(diff_izq_eog, s_SRate, 0.08)
    diff_der_eog_avg = fn.f_AvFlt(diff_der_eog, s_SRate, 0.08)
    #
    # # Maximo de ventana
    #
    # diff_izq_emg_avg = np.max(sig_izq_emg)
    # diff_der_emg_avg = np.max(sig_der_emg)
    # diff_arr_emg_avg = np.max(sig_arr_emg)
    #
    # # Movimiento EOG
    #
    # mov = fn.identificar_movimiento(diff_der_eog_avg, diff_izq_eog_avg, U_Derecha_EOG, U_Izquierda_EOG)
    # Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)
    #
    # # Movimiento EMG
    #
    # if diff_arr_emg_avg > U_Arriba and not diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG:
    #     mov_emg = 'MF'
    #     print(mov_emg)
    #     Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)
    #
    # elif diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG and not diff_arr_emg_avg > U_Arriba:
    #     mov_emg = 'CD'
    #     print(mov_emg)
    #     Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)
    #
    # elif diff_izq_emg_avg > U_Izquierda_EMG and not diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
    #     mov_emg = 'CI'
    #     print(mov_emg)
    #     Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)
    #
    # elif diff_izq_emg_avg > U_Izquierda_EMG and diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
    #     mov_emg = 'C'
    #     print(mov_emg)
    #     Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)
    #
    # else:
    #     mov_emg = 'Nada'

    return np.array([sig_arr_emg, sig_der_emg, sig_izq_eog, diff_der_eog_avg, diff_izq_eog_avg])


data_pr = procesamiento(Temp1)

# FFTs = np.fft.rfft(Temp1.T)
# FFTs_freqs = np.fft.rfftfreq(len(Temp1), d=1./s_SRate)

##
plt.figure()
plt.plot(data_pr[3], 'b')
plt.plot(data_pr[4], 'r')
plt.title('Señal procesada')
plt.xlabel('No. Muestra')
plt.ylabel('Amplitud [mV]')


#hola
##

