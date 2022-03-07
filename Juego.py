##
import Funciones as fn
from f_SignalProcFuncLibs import *
import scipy.signal as sig
from pyOpenBCI import OpenBCICyton
import pygame
import pyautogui as gui

calibracion = input('¿Desea Calibrar? (si/no)')


def adquisicion_cal(sample):
    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    if len(inc_data) == 1700:
        board.stop_stream()

if calibracion == 'si':

    Tipo_Señal = input('Indique el tipo de señal (EMG/EOG)')
    Tipo_Movimiento = input('Indique el movimiento a realizar')

    s_SRate = 250  # Hertz

    inc_data = []

    board = OpenBCICyton(port='COM3')
    uVolts_per_count = (4500000) / 24 / (2 ** 23 - 1)  # uV/count

    # board.write_command('1')
    # board.write_command('2')
    # board.write_command('3')
    board.write_command('4')
    # board.write_command('5')
    # board.write_command('6')
    board.write_command('7')
    board.write_command('8')

    board.start_stream(adquisicion_cal)

    Umbral = fn.Calibracion(inc_data, Tipo_Señal, Tipo_Movimiento)


    if Tipo_Señal == 'EOG':
        if Tipo_Movimiento == 'Parpadeo':
            np.savetxt('U_Parpadeo.txt', Umbral)
            U_Parpadeo = Umbral

        elif Tipo_Movimiento == 'Derecha':
            np.savetxt('U_Derecha_EOG.txt', Umbral)
            U_Derecha_EOG = Umbral

        elif Tipo_Movimiento == 'Izquierda':
            np.savetxt('U_Izquierda_EOG.txt', Umbral)
            U_Izquierda_EOG = Umbral
        else:
            print('Palabra incorrecta')

    elif Tipo_Señal == 'EMG':
        if Tipo_Movimiento == 'Arriba':
            Umbral = np.array([Umbral, 0])
            np.savetxt('U_Arriba.txt', Umbral)
            U_Arriba = Umbral[0]

        elif Tipo_Movimiento == 'Derecha':
            Umbral = np.array([Umbral, 0])
            np.savetxt('U_Derecha_EMG.txt', Umbral)
            U_Derecha_EMG = Umbral[0]


        elif Tipo_Movimiento == 'Izquierda':
            Umbral = np.array([Umbral, 0])
            np.savetxt('U_Izquierda_EMG.txt')
            U_Izquierda_EMG = Umbral[0]

    else:
        print('Palabra Incorrecta')

elif calibracion == 'no':

    U_Parpadeo = np.array(np.loadtxt('U_Parpadeo.txt'))
    U_Derecha_EOG = np.array(np.loadtxt('U_Derecha_EOG.txt'))
    U_Izquierda_EOG = np.array(np.loadtxt('U_Izquierda_EOG.txt'))

    U_Arriba = np.array(np.loadtxt('U_Arriba.txt'))
    U_Arriba = U_Arriba[0]
    U_Izquierda_EMG = np.array(np.loadtxt('U_Izquierda_EMG.txt'))
    U_Izquierda_EMG = U_Izquierda_EMG[0]

    U_Derecha_EMG = np.array(np.loadtxt('U_Derecha_EMG.txt'))
    U_Derecha_EMG = U_Derecha_EMG[0]


## Definición de la clase
class pre_wind:
    def __init__(self):
        self.data = []
        self.count = 0
        self.e1 = 'Nada'
        self.e2 = 'Nada'

    def actualizar(self, Var = 'Nada'):
        self.count += 1
        if Var != 'Nada':
            if self.e1 == 'Nada':
                self.e1 = Var
                self.count = 0
            else:
                self.e2 = Var

        if self.e2 != 'Nada':

            if self.e1 == 'Parpadeo' and self.e2 == 'Derecha':
                pygame.mixer.init()
                pygame.mixer.music.load("p1.mp3")
                pygame.mixer.music.play()

            elif self.e1 == 'Parpadeo' and self.e2 == 'Izquierda':
                pygame.mixer.init()
                pygame.mixer.music.load("p2.mp3")
                pygame.mixer.music.play()

            elif self.e1 == 'Derecha' and self.e2 == 'Derecha':
                pygame.mixer.init()
                pygame.mixer.music.load("p3.mp3")
                pygame.mixer.music.play()

            self.count = 0
            self.e1 = 'Nada'
            self.e2 = 'Nada'

        if self.count == 10:

            if self.e1 != 'Nada':

                if self.e1 == 'Parpadeo':

                    pygame.mixer.init()
                    pygame.mixer.music.load("p4.mp3")
                    pygame.mixer.music.play()

                elif self.e1 == 'Derecha':
                    pygame.mixer.init()
                    pygame.mixer.music.load("p5.mp3")
                    pygame.mixer.music.play()

                elif self.e1 == 'Izquierda':
                    pygame.mixer.init()
                    pygame.mixer.music.load("p6.mp3")
                    pygame.mixer.music.play()


            self.count = 0
            self.e1 = 'Nada'
            self.e2 = 'Nada'

## Constantes

s_SRate = 250 # Hertz
window = 1 # segundos
act = 0.25 # segundos

## Filtros

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

##

class proc_wind:
    def __init__(self, channels, window, act):
        self.data = np.zeros((window, channels))
        self.inc_lenght = act

    def refresh(self, inc_data):
        ind = self.inc_lenght
        temp = self.data[ind:, :]
        self.data[:-ind, :] = temp
        self.data[-ind:, :] = np.array(inc_data)

class c_keys:
    def __init__(self):
        self.arr = 'up'
        self.ab = 'down'
        self.izq = 'left'
        self.der = 'right'
        self.intro = 'enter'

    def change(self):
        if self.arr == 'up':
            self.arr = 'w'
            self.ab = 's'
            self.izq = 'a'
            self.der = 'd'
            self.intro = ' '
        elif self.arr == 'w':
            self.arr = 'up'
            self.ab = 'down'
            self.izq = 'left'
            self.der = 'right'
            self.intro = 'enter'

def adquisicion(sample):
    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    if len(inc_data) == int(s_SRate * act):
        ventana.refresh(inc_data)
        procesamiento(ventana.data)
        inc_data.clear()

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

    if Mov == 'Derecha':
        eog = True
        gui.keyDown(keys.intro)
        print('EOG derecha')
    else:
        eog = False
    if diff_arr_emg_avg > U_Arriba and not diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG:
        arriba1 = True
        gui.keyDown(keys.arr)
        print('EMG arriba')
    else:
        arriba1 = False
    if diff_der_emg_avg > U_Derecha_EMG and not diff_izq_emg_avg > U_Izquierda_EMG and not diff_arr_emg_avg > U_Arriba:
        derecha1 = True
        gui.keyDown(keys.der)
        print('EMG derecha')
    else:
        derecha1 = False
    if diff_izq_emg_avg > U_Izquierda_EMG and not diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
        izquierda1 = True
        gui.keyDown(keys.izq)
        print('EMG izquierda')
    else:
        izquierda1 = False
    if diff_izq_emg_avg > U_Izquierda_EMG and diff_der_emg_avg > U_Derecha_EMG and not diff_arr_emg_avg > U_Arriba:
        abajo1 = True
        gui.keyDown(keys.ab)
        print('EMG abajo')
    else:
        abajo1 = False
    if diff_izq_emg_avg > U_Izquierda_EMG and diff_der_emg_avg > U_Derecha_EMG and diff_arr_emg_avg > U_Arriba:
         todos1 = True
         keys.change()
         print('EMG todos', keys.arr, keys.ab, keys.der, keys.izq, keys.intro)

ventana = proc_wind(8, int(s_SRate * window), int(act * s_SRate))
Movimiento = pre_wind()
keys = c_keys()
inc_data = []

board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000) / 24 / (2 ** 23 - 1)  # uV/count

# board.write_command('1')
# board.write_command('2')
# board.write_command('3')
board.write_command('4')
# board.write_command('5')
# board.write_command('6')
board.write_command('7')
board.write_command('8')

board.start_stream(adquisicion)