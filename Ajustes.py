##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Funciones as fn
import scipy.signal as sig
from f_SignalProcFuncLibs import *

#Constantes
s_SRate = 250  # Hertz
window = 1 # segundos
act = 0.25 # segundos

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 50], [15, 52])



Temp_read1 = pd.read_csv(os.path.join('initial_tests', 'parpadeo_nico.txt'), header=4)
# Temp_read1 = pd.read_csv(os.path.join('initial_tests', 'OpenBCI-RAW-2022-05-10_20-37-00.txt'), header=4)
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
    sig_izq_eog_avg = fn.f_AvFlt(sig_izq_eog, s_SRate, 0.2)
    sig_der_eog_avg = fn.f_AvFlt(sig_der_eog, s_SRate, 0.2)

    # Primera derivada

    diff_izq_eog = np.diff(sig_izq_eog_avg)
    diff_der_eog = np.diff(sig_der_eog_avg)

    # Suavizado 2

    diff_izq_eog_avg = fn.f_AvFlt(diff_izq_eog, s_SRate, 0.2)
    diff_der_eog_avg = fn.f_AvFlt(diff_der_eog, s_SRate, 0.2)
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


plt.figure()
plt.plot(data_pr[3], 'b')
plt.plot(data_pr[4], 'r')
plt.title('Señal procesada')
plt.xlabel('No. Muestra')
plt.ylabel('Amplitud [mV]')

##
def Calcular_Umbral (data, tipo):

    if tipo == 'EOG':
        a = 5

        vent_der = data[0]
        vent_izq = data[1]

        picos_der, n = sig.find_peaks(np.abs(vent_der), height=7)
        picos_izq, n = sig.find_peaks(np.abs(vent_izq), height=7)

        Der = np.array([vent_der[picos_der]])
        Izq = np.array([vent_izq[picos_izq]])

        der_pos = Der[Der > 0]
        U_pos_der = np.mean(der_pos) - a

        der_neg = Der[Der < 0]
        U_neg_der = np.mean(der_neg) + a

        izq_pos = Izq[Izq > 0]
        U_pos_izq = np.mean(izq_pos) - a

        izq_neg = Izq[Izq < 0]
        U_neg_izq = np.mean(izq_neg) + a

        return U_pos_der, U_pos_izq, U_neg_der, U_neg_izq

    elif tipo == 'EMG':

        picos, n = sig.find_peaks(data, height=50)
        Umbral = np.mean(data[picos])

        return Umbral


# Función principal de calibración
def Calibracion(inc_data, Tipo_Señal, Tipo_Movimiento, s_SRate = 250):

    inc_data = np.array(inc_data)

    filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
    filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

    if Tipo_Señal == 'EOG':

        sig_der_eog = sig.detrend(inc_data[3][500:])
        sig_izq_eog = sig.detrend(inc_data[4][500:])

        sig_izq_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_izq_eog)
        sig_der_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_der_eog)

        sig_izq_eog_avg = fn.f_AvFlt(sig_izq_eog, s_SRate, 0.08)
        sig_der_eog_avg = fn.f_AvFlt(sig_der_eog, s_SRate, 0.08)

        diff_izq_eog = np.diff(sig_izq_eog_avg)
        diff_der_eog = np.diff(sig_der_eog_avg)

        diff_izq_eog_avg = fn.f_AvFlt(diff_izq_eog, s_SRate, 0.08)
        diff_der_eog_avg = fn.f_AvFlt(diff_der_eog, s_SRate, 0.08)

        data = np.array([diff_der_eog_avg, diff_izq_eog_avg])

        U_pos_der, U_pos_izq, U_neg_der, U_neg_izq = Calcular_Umbral(data, 'EOG')

        if Tipo_Movimiento == 'Parpadeo':
            U_Parpadeo = [U_pos_der, U_pos_izq, U_neg_der, U_neg_izq]
            return U_Parpadeo

        elif Tipo_Movimiento == 'Derecha':
            U_Derecha_EOG = [U_pos_der, U_pos_izq, U_neg_der, U_neg_izq]
            return U_Derecha_EOG

        elif Tipo_Movimiento == 'Izquierda':
            U_Izquierda_EOG = [U_pos_der, U_pos_izq, U_neg_der, U_neg_izq]
            return U_Izquierda_EOG

        else:
            print('Palabra Incorrecta')

    elif Tipo_Señal == 'EMG':

        if Tipo_Movimiento == 'Arriba':

            sig_arr_emg = sig.detrend(inc_data[500:, 0])
            sig_arr_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_arr_emg)
            sig_arr_emg = np.abs(sig_arr_emg)

            U_Arriba_EMG = Calcular_Umbral(sig_arr_emg, 'EMG')
            return U_Arriba_EMG


        elif Tipo_Movimiento == 'Derecha':
            sig_der_emg = sig.detrend(inc_data[500:, 1])
            sig_der_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_der_emg)
            sig_der_emg = np.abs(sig_der_emg)

            U_Derecha_EMG = Calcular_Umbral(sig_der_emg, 'EMG')
            return U_Derecha_EMG

        elif Tipo_Movimiento == 'Izquierda':

            sig_izq_emg = sig.detrend(inc_data[500:, 2])
            sig_izq_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_izq_emg)
            sig_izq_emg = np.abs(sig_izq_emg)

            U_Izquierda_EMG = Calcular_Umbral(sig_izq_emg, 'EMG')
            return U_Izquierda_EMG

        else:
            print('Palabra Incorrecta')

    else:
        print('Palabra Incorrecta')



##

derecha1 = data_pr[3][1600:2200]
derecha2 = data_pr[4][1600:2200]

plt.figure()
plt.plot(derecha1, 'b')
plt.plot(derecha2, 'r')
plt.title('Señal procesada')
plt.xlabel('No. Muestra')
plt.ylabel('Amplitud [mV]')

