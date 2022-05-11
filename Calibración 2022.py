##
import Funciones as fn
from pyOpenBCI import OpenBCICyton
import numpy as np
import pygame

##
Tipo_Señal = input('Indique el tipo de señal') #EOG/EMG
Tipo_Movimiento = input('Indique el movimiento a realizar') #EOG: Parpadeo, Derecha, Izquierda.
                                                            #EMG: Arriba, Derecha, izquierda

#Constantes
s_SRate = 250  # Hertz
board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000) / 24 / (2 ** 23 - 1)  # uV/count

inc_data = []

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

    if len(inc_data) == 1700:
        board.stop_stream()

if mov == 'Parpadeo':


pygame.mixer.init()
pygame.mixer.music.load("mp3//demo.mp3")
pygame.mixer.music.play()
time.sleep(1.5)
fn.play_vid(mov)
time.sleep(0.5)

board.start_stream(adquisicion_cal)

# Calculo de umbral
Umbral = fn.Calibracion(inc_data, Tipo_Señal, Tipo_Movimiento)

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

