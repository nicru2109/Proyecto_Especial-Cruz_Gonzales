##
import Funciones as fn
from pyOpenBCI import OpenBCICyton
import numpy as np
import pygame
import time
from Clases import *
from f_SignalProcFuncLibs import *
import Funciones as fn
import scipy.signal as sig
import matplotlib.pyplot as plt

##

#Constantes
s_SRate = 250 # Hertz
window = 1 # segundos
act = 0.25 # segundos
tspan_exps = 25 * s_SRate

board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000) / 24 / (2 ** 23 - 1)  # uV/count

# Filtros
filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

Movimiento = pre_wind()
ventana = proc_wind(8, int(s_SRate * window), int(act * s_SRate))

#Apagar canales 4, 7 y 8

# board.write_command('1')
# board.write_command('2')
# board.write_command('3')
board.write_command('4')
# board.write_command('5')
# board.write_command('6')
board.write_command('7')
board.write_command('8')

# Toma de datos
def adquisicion_cal(sample):
    if len(inc_data) == 0:
        pygame.mixer.init()
        pygame.mixer.music.load("mp3//go.mp3")
        pygame.mixer.music.play()

    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    # if (len(inc_data) % int(s_SRate * act)) == 0:
    #     ventana.refresh(inc_data[-int(s_SRate * act):])
    #     procesamiento(ventana.data)

    if len(inc_data) == 2000:
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

    pr_data.append(np.array([sig_arr_emg[-int(s_SRate * act):], sig_der_emg[-int(s_SRate * act):], sig_izq_emg[-int(s_SRate * act):], diff_der_eog_avg[-int(s_SRate * act):], diff_izq_eog_avg[-int(s_SRate * act):]]))

nombre = input('Indique el nombre del archivo')

while 1:

    inc_data = []
    pr_data = []

    Tipo_Señal = input('Indique el tipo de señal')  # EOG/EMG
    Tipo_Movimiento = input('Indique el movimiento a realizar')  # EOG: Parpadeo, Derecha, Izquierda.
    # EMG: Arriba, Derecha, izquierda

    if Tipo_Movimiento == 'Parpadeo':
        mov = 'P'
    elif Tipo_Movimiento == 'Derecha' and Tipo_Señal == 'EOG':
        mov = 'D'
    elif Tipo_Movimiento == 'Izquierda' and Tipo_Señal == 'EOG':
        mov = 'I'
    elif Tipo_Movimiento == 'Izquierda':
        mov = 'CI'
    elif Tipo_Movimiento == 'Derecha':
        mov = 'CD'
    elif Tipo_Movimiento == 'Arriba':
        mov = 'MF'


    pygame.mixer.init()
    pygame.mixer.music.load("mp3//demo.mp3")
    pygame.mixer.music.play()
    time.sleep(1.5)
    fn.play_vid(mov)
    time.sleep(0.5)
    pygame.mixer.music.load("mp3//prep.mp3")
    pygame.mixer.music.play()
    time.sleep(3)

    board.start_stream(adquisicion_cal)

    l = len(pr_data)
    z = pr_data[0].shape[1]

    data = np.zeros((pr_data[0].shape[0], l * z))

    for i in range(l):
        data[:, (i * z):((i + 1) * z)] = pr_data[i]

    # Calculo de umbral
    Umbral = fn.Calibracion_ventana(data, Tipo_Señal, Tipo_Movimiento)

    #Guardar umbral en variable
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

    else:
        print('Palabra Incorrecta')

    print('Umbral Calibrado Correctamente')

    np.savetxt('initial_tests//' + nombre + '.txt', data)

    end = input('Desea terminar? (si/no)')

    if end == 'si':
        break


## Umbrales

U_Parpadeo = np.array(np.loadtxt('trhlds//U_Parpadeo.txt'))
U_Derecha_EOG = np.array(np.loadtxt('trhlds//U_Derecha_EOG.txt'))
U_Izquierda_EOG = np.array(np.loadtxt('trhlds//U_Izquierda_EOG.txt'))

## Carga y procesamiento

Temp1 = np.array(np.loadtxt('initial_tests//' + nombre + '.txt'))
Temp1 = Temp1[:, 250:]

plt.figure()
plt.plot(Temp1[3], 'b')
plt.plot(Temp1[4], 'r')
plt.axline(xy1=[0, U_Parpadeo[0]], slope=0, color='b')
plt.axline(xy1=[0, U_Parpadeo[1]], slope=0, color='r')
plt.axline(xy1=[0, U_Parpadeo[2]], slope=0, color='b')
plt.axline(xy1=[0, U_Parpadeo[3]], slope=0, color='r')
plt.title('Señal procesada')
plt.xlabel('No. Muestra')
plt.ylabel('Amplitud [mV]')
