## Imports
import Funciones as fn
import numpy as np
import scipy.signal as sig
from f_SignalProcFuncLibs import *
from Clases import *
import matplotlib.pyplot as plt

##
s_SRate = 250  # Hertz
window = 1 # segundos
act = 0.25 # segundos
filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 50], [15, 52])

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

    return np.array([sig_arr_emg[1:], sig_der_emg[1:], sig_izq_emg[1:], diff_der_eog_avg, diff_izq_eog_avg])


def graficador(path):

    data = np.array(np.loadtxt('initial_tests//' + path + '.txt'))
    data = data[250:, :]

    fig, axs = plt.subplots(1, 2)

    fig.suptitle('Señal cruda', fontsize=16)

    axs[0].plot(data[1:, 4], 'b')
    axs[0].set_xlabel('No. Muestra')
    axs[0].set_ylabel('Amplitud [mV]')
    axs[0].set_title('Derecho')

    axs[1].plot(data[1:, 4], 'r')
    axs[1].set_xlabel('No. Muestra')
    axs[1].set_ylabel('Amplitud [mV]')
    axs[1].set_title('Izquierdo')

    fft_der = np.abs(np.fft.rfft(data[1:, 4]))
    fft_izq = np.abs(np.fft.rfft(data[1:, 5]))

    fft_freqs = np.fft.rfftfreq(len(data[1:, 4]), 1 / 250)

    fig, axs = plt.subplots(1, 2)

    fig.suptitle('Espectro de frecuencias', fontsize=16)

    axs[0].plot(fft_freqs, fft_der, 'b')
    axs[0].set_xlabel('Hertz [Hz]')
    axs[0].set_ylabel('Magnitud')
    axs[0].set_title('Derecho')
    axs[0].set_ylim((0, 30000))

    axs[1].plot(fft_freqs, fft_der, 'r')
    axs[1].set_xlabel('Hertz [Hz]')
    axs[1].set_ylabel('Magnitud')
    axs[1].set_title('Izquierdo')
    axs[1].set_ylim((0, 30000))

    data_pr = procesamiento(data)

    fig, axs = plt.subplots(1, 2)

    fig.suptitle('Señal procesada\nPrimera derivada', fontsize=16)

    axs[0].plot(data_pr[3], 'b')
    axs[0].set_xlabel('No. Muestra')
    axs[0].set_ylabel('Magnitud')
    axs[0].set_title('Derecho')

    axs[1].plot(data_pr[4], 'r')
    axs[1].set_xlabel('No. Muestra')
    axs[1].set_ylabel('Magnitud')
    axs[1].set_title('Izquierdo')