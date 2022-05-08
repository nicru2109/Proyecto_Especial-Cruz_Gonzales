##
import Funciones as fn
from f_SignalProcFuncLibs import *
import scipy.signal as sig
from pyOpenBCI import OpenBCICyton
import matplotlib.pyplot as plt
import numpy as np

s_SRate = 250 # Hertz
inc_data = []



##

Tipo_Señal = input('Indique la señal (EOG o EMG): ')
Tipo_Movimiento = input('Indique el movimiento: ')

board = OpenBCICyton(port='COM3')
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count

#board.write_command('1')
#board.write_command('2')
#board.write_command('3')
board.write_command('4')
#board.write_command('5')
#board.write_command('6')
board.write_command('7')
board.write_command('8')


def adquisicion_cal(sample):
    inc_data.append(np.array(sample.channels_data) * uVolts_per_count)

    if len(inc_data) == 1700:
        board.stop_stream()

board.start_stream(adquisicion_cal)



##
inc_data = np.array(inc_data)

filt_FiltSOS_eog = f_GetIIRFilter(s_SRate, [0.015, 10], [0.01, 12])
filt_FiltSOS_emg = f_GetIIRFilter(s_SRate, [20, 57], [15, 59])

if Tipo_Señal == 'EOG':

    sig_der_eog = sig.detrend(inc_data[500:, 4])
    sig_izq_eog = sig.detrend(inc_data[500:, 5])

    sig_izq_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_izq_eog)
    sig_der_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_der_eog)

    sig_izq_eog_avg = fn.f_AvFlt(sig_izq_eog, s_SRate, 0.08)
    sig_der_eog_avg = fn.f_AvFlt(sig_der_eog, s_SRate, 0.08)

    diff_izq_eog = np.diff(sig_izq_eog_avg)
    diff_der_eog = np.diff(sig_der_eog_avg)

    diff_izq_eog_avg = fn.f_AvFlt(diff_izq_eog, s_SRate, 0.08)
    diff_der_eog_avg = fn.f_AvFlt(diff_der_eog, s_SRate, 0.08)

    data = np.array([diff_der_eog_avg, diff_izq_eog_avg])

    U_pos_der, U_pos_izq, U_neg_der, U_neg_izq = fn.Calibrar_umbrales(data, 'EOG')

    if Tipo_Movimiento == 'Parpadeo':
        U_Parpadeo = [U_pos_der, U_pos_izq, U_neg_der, U_neg_izq]

    elif Tipo_Movimiento == 'Derecha':
        U_Derecha_EOG = [U_pos_der, U_pos_izq, U_neg_der, U_neg_izq]

    elif Tipo_Movimiento == 'Izquierda':
        U_Izquierda_EOG = [U_pos_der, U_pos_izq, U_neg_der, U_neg_izq]

    else:
        print('Inconcluso')

elif Tipo_Señal == 'EMG':

    if Tipo_Movimiento == 'Arriba':

        sig_arr_emg = sig.detrend(inc_data[500:, 0])
        sig_arr_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_arr_emg)
        sig_arr_emg = np.abs(sig_arr_emg)

        U_Arriba_EMG = fn.Calibrar_umbrales(sig_arr_emg, 'EMG')


    elif Tipo_Movimiento == 'Derecha':
        sig_der_emg = sig.detrend(inc_data[500:, 1])
        sig_der_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_der_emg)
        sig_der_emg = np.abs(sig_der_emg)

        U_Derecha_EMG = fn.Calibrar_umbrales(sig_der_emg, 'EMG')

    elif Tipo_Movimiento == 'Izquierda':

        sig_izq_emg = sig.detrend(inc_data[500:, 2])
        sig_izq_emg = signal.sosfiltfilt(filt_FiltSOS_emg, sig_izq_emg)
        sig_izq_emg = np.abs(sig_izq_emg)

        U_Izquierda_EMG = fn.Calibrar_umbrales(sig_izq_emg, 'EMG')

    else:
        print('Inconcluso')

else:
    print('Inconcluso')

##
# plt.figure()
# plt.plot(diff_izq_eog_avg, 'b')
# plt.plot(diff_der_eog_avg, 'r')
# plt.title('Señal procesada')
# plt.xlabel('No. Muestra')
# plt.ylabel('Amplitud [mV]')
#




