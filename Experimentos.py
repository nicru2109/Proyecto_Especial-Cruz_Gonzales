## Imports
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

#Definir la señal a ejecutar
##
sujeto = input('Indique el nombre de la sesión ')
Tipo_Señal = input('Indique el tipo de señal (EOG o EMG) ')
mode = input(('Indique el tipo de experimento ')) #train/test

## Constantes

s_SRate = 250 # Hertz
window = 1 # segundos
act = 0.25 # segundos
tspan_exps = 5 * s_SRate

# Configuración de la board

board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count

# Filtros

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

# Lista inicial
inc_data = []

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

#
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
        inc_data.clear()
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

    # Maximo de ventana

    diff_izq_emg_avg = np.max(sig_izq_emg)
    diff_der_emg_avg = np.max(sig_der_emg)
    diff_arr_emg_avg = np.max(sig_arr_emg)

    # Movimiento EOG

    mov = fn.identificar_movimiento(diff_der_eog_avg, diff_izq_eog_avg, U_Derecha_EOG, U_Izquierda_EOG)
    Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)

    # Movimiento EMG

    if diff_arr_emg_avg > U_Arriba and not diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG:
        mov_emg = 'MF'
        print(mov_emg)
        Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)

    elif diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG and not diff_arr_emg_avg > U_Arriba:
        mov_emg = 'CD'
        print(mov_emg)
        Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)

    elif diff_izq_emg_avg > U_Izquierda_EMG and not diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
        mov_emg = 'CI'
        print(mov_emg)
        Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)

    elif diff_izq_emg_avg > U_Izquierda_EMG and diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
        mov_emg = 'C'
        print(mov_emg)
        Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)

    else:
        mov_emg = 'Nada'
        Movimiento.actualizar(mov, mode=mode, sig_type= Tipo_Señal)



## Definición de diccionarios

dic_EMG = {'Mov1': 'MF', 'Mov2': 'CI', 'Mov3': 'CD', 'Mov4': 'C'}
dic_EOG = {'Mov1': 'P', 'Mov2': 'PP', 'Mov3': 'PI', 'Mov4': 'PD',
           'Mov5': 'IP', 'Mov6': 'DP', 'Mov7': 'I', 'Mov8': 'D'}

##

if mode == 'test':
    if Tipo_Señal == "EOG":
        movs_list = fn.mov_list(mode=mode, **dic_EOG)

    elif Tipo_Señal == "EMG" :
        movs_list = fn.mov_list(mode=mode, **dic_EMG)

if mode == 'train':
    if Tipo_Señal == "EOG":
        movs_list = fn.mov_list(mode=mode, **dic_EOG)

    elif Tipo_Señal == "EMG" :
        movs_list = fn.mov_list(mode=mode, **dic_EMG)

Movimiento = pre_wind()
ventana = proc_wind(8, int(s_SRate * window), int(act * s_SRate))

fn.play_vid('C')

for i in range(len(movs_list)):

    pygame.mixer.init()
    pygame.mixer.music.load("mp3//demo.mp3")
    pygame.mixer.music.play()

    time.sleep(1.5)

    fn.play_vid(movs_list[i])

    time.sleep(0.5)

    pygame.mixer.music.load("mp3//prep.mp3")
    pygame.mixer.music.play()

    time.sleep(3)



    board.start_stream(adquisicion)

    if len(Movimiento.array) < i + 1:
        Movimiento.add_null()

result = {'Anotacion': movs_list, 'Resultado': Movimiento.array}

df_result = pd.DataFrame(result)

df_result.to_csv('resultados//' + sujeto + '.csv')



# Inicio de toma de datos
