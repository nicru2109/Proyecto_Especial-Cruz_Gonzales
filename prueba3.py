# Primer script para las pruebas de rendimiento de la interfaz

## Imports
from Clases import *
from f_SignalProcFuncLibs import *
import numpy as np
import Funciones as fn
from pyOpenBCI import OpenBCICyton
import scipy.signal as sig
import time

## Funciones


# Funcion de adquisición de la calibración
def adquisicion_cal(sample):
    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    if len(inc_data) == 1700:
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
    sig_izq_eog_avg = fn.f_AvFlt(sig_izq_eog, s_SRate, 0.08)
    sig_der_eog_avg = fn.f_AvFlt(sig_der_eog, s_SRate, 0.08)

    # Primera derivada

    diff_izq_eog = np.diff(sig_izq_eog_avg)
    diff_der_eog = np.diff(sig_der_eog_avg)

    # Suavizado 2

    diff_izq_eog_avg = fn.f_AvFlt(diff_izq_eog, s_SRate, 0.08)
    diff_der_eog_avg = fn.f_AvFlt(diff_der_eog, s_SRate, 0.08)

    # Maximo de ventana

    diff_izq_emg_avg = np.max(sig_izq_emg)
    diff_der_emg_avg = np.max(sig_der_emg)
    diff_arr_emg_avg = np.max(sig_arr_emg)

    # Movimiento EOG

    Mov = fn.identificar_movimiento(diff_der_eog_avg, diff_izq_eog_avg, U_Derecha_EOG, U_Izquierda_EOG)
    print(Mov)

    Movimiento.actualizar(Mov)

# Función sesiones dirigidas
def sesiones(movs_list, mode='train'):

    for mov in movs_list:

        pygame.mixer.init()
        pygame.mixer.music.load("mp3//demo.mp3")
        pygame.mixer.music.play()


        fn.play_vid(mov)

        time.sleep(0.5)

        pygame.mixer.music.load("mp3//prep.mp3")
        pygame.mixer.music.play()

        time.sleep(3)

        if mode == 'train':

            time.sleep(3)

        elif mode == 'test':

            time.sleep(3)

# Función para mostrar videos

##

calibracion = input('¿Desea Calibrar? (si/no)')


if calibracion == 'si':

    Tipo_Señal = input('Indique el tipo de señal (EMG/EOG)')
    Tipo_Movimiento = input('Indique el movimiento a realizar')

    s_SRate = 250  # Hertz

    inc_data = []

    board = OpenBCICyton(port='COM3')
    uVolts_per_count = (4500000) / 24 / (2 ** 23 - 1)  # uV/count

    board.write_command('4')
    board.write_command('7')
    board.write_command('8')

    board.start_stream(adquisicion_cal)

    Umbral = fn.Calibracion(inc_data, Tipo_Señal, Tipo_Movimiento)


    if Tipo_Señal == 'EOG':
        if Tipo_Movimiento == 'Parpadeo':
            np.savetxt('trhlds//U_Parpadeo.txt', Umbral)
            U_Parpadeo = Umbral

        elif Tipo_Movimiento == 'Derecha':
            np.savetxt('trhlds//U_Derecha_EOG.txt', Umbral)
            U_Derecha_EOG = Umbral

        elif Tipo_Movimiento == 'Izquierda':
            np.savetxt('trhlds//U_Izquierda_EOG.txt', Umbral)
            U_Izquierda_EOG = Umbral
        else:
            print('Palabra incorrecta')

    elif Tipo_Señal == 'EMG':
        if Tipo_Movimiento == 'Arriba':
            Umbral = np.array([Umbral, 0])
            np.savetxt('trhlds//U_Arriba.txt', Umbral)
            U_Arriba = Umbral[0]

        elif Tipo_Movimiento == 'Derecha':
            Umbral = np.array([Umbral, 0])
            np.savetxt('trhlds//U_Derecha_EMG.txt', Umbral)
            U_Derecha_EMG = Umbral[0]


        elif Tipo_Movimiento == 'Izquierda':
            Umbral = np.array([Umbral, 0])
            np.savetxt('trhlds//U_Izquierda_EMG.txt', Umbral)
            U_Izquierda_EMG = Umbral[0]

    else:
        print('Palabra Incorrecta')

    print('Umbral Calibrado Correctamente')

elif calibracion == 'no':

    U_Parpadeo = np.array(np.loadtxt('trhlds//U_Parpadeo.txt'))
    U_Derecha_EOG = np.array(np.loadtxt('trhlds//U_Derecha_EOG.txt'))
    U_Izquierda_EOG = np.array(np.loadtxt('trhlds//U_Izquierda_EOG.txt'))

    U_Arriba = np.array(np.loadtxt('trhlds//U_Arriba.txt'))
    U_Arriba = U_Arriba[0]

    U_Izquierda_EMG = np.array(np.loadtxt('trhlds//U_Izquierda_EMG.txt'))
    U_Izquierda_EMG = U_Izquierda_EMG[0]

    U_Derecha_EMG = np.array(np.loadtxt('trhlds//U_Derecha_EMG.txt'))
    U_Derecha_EMG = U_Derecha_EMG[0]

## Definición de diccionarios

dic_EMG = {'Mov1': 'MF', 'Mov2': 'CI', 'Mov3': 'CD', 'Mov4': 'C'}
dic_EOG = {'Mov1': 'P', 'Mov2': 'PP', 'Mov3': 'PI', 'Mov4': 'PD',
           'Mov5': 'IP', 'Mov6': 'DP', 'Mov7': 'I', 'Mov8': 'D'}

## Constantes y configuración inicial

s_SRate = 250 # Hertz
window = 1 # segundos
act = 0.25 # segundos

board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

inc_data = []

##

# Pruebas EOG


