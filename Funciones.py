# Funciones
from f_SignalProcFuncLibs import *
import numpy as np
import random
import scipy.signal as sig
import cv2

# Función filtro promedio
def f_AvFlt(data, w, t):

    data = data.squeeze()
    temp = np.zeros(np.size(data))

    wind = w*t
    for i in range(0, len(temp)):

        ini = round(i - wind/2)
        fin = round(i + wind/2)

        if ini < 0:

            ini = 0
        if fin > len(data):

            fin = len(data)

        w_temp = data[ini:fin]
        mean = np.mean(w_temp)

        temp[i] = mean

    return temp

# Función para calcular umbral en la etapa de calibración
def Calcular_Umbral (data, tipo):

    if tipo == 'EOG':
        a = 5

        vent_der = data[0, :]
        vent_izq = data[1, :]

        picos_der, n = sig.find_peaks(np.abs(vent_der), height=7)
        picos_izq, n = sig.find_peaks(np.abs(vent_izq), height=7)

        Der = np.array([vent_der[picos_der]])
        Izq = np.array([vent_izq[picos_izq]])

        der_pos = Der[Der > 0]
        U_pos_der = (np.mean(der_pos) - a) + (np.mean(der_pos) - a)*0.15

        der_neg = Der[Der < 0]
        U_neg_der = (np.mean(der_neg) + a) + (np.mean(der_neg) + a)*0.15

        izq_pos = Izq[Izq > 0]
        U_pos_izq = (np.mean(izq_pos) - a) + (np.mean(izq_pos) - a)*0.15

        izq_neg = Izq[Izq < 0]
        U_neg_izq = (np.mean(izq_neg) + a) + (np.mean(izq_neg) + a)*0.15

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

        sig_der_eog = sig.detrend(inc_data[500:, 4])
        sig_izq_eog = sig.detrend(inc_data[500:, 5])

        sig_izq_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_izq_eog)
        sig_der_eog = signal.sosfiltfilt(filt_FiltSOS_eog, sig_der_eog)

        sig_izq_eog_avg = f_AvFlt(sig_izq_eog, s_SRate, 0.2)
        sig_der_eog_avg = f_AvFlt(sig_der_eog, s_SRate, 0.2)

        diff_izq_eog = np.diff(sig_izq_eog_avg)
        diff_der_eog = np.diff(sig_der_eog_avg)

        diff_izq_eog_avg = f_AvFlt(diff_izq_eog, s_SRate, 0.2)
        diff_der_eog_avg = f_AvFlt(diff_der_eog, s_SRate, 0.2)

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

# Identificar el patrón individual de EOG
def identificar_movimiento(ventana_der,ventana_izq, U_Derecha, U_Izquierda):

    #U_variable son arreglos con umbral de señal derecha positiva, izquierda positiva, derecha negativa
    #e izquierda negativa, en ese orden

    Mov = 'Nada'

    picos_der, n = sig.find_peaks(np.abs(ventana_der), height=2)
    picos_izq, n = sig.find_peaks(np.abs(ventana_izq), height=2)

    if len(picos_izq) == 0 or len(picos_der) == 0:
        Mov = 'Nada'

    elif ventana_der[picos_der[0]] < -4 and ventana_izq[picos_izq[0]] < -4 \
            and ventana_der[picos_der[-1]] > 4 and ventana_izq[picos_izq[-1]] > 4:
        Mov = 'P'

    elif ventana_der[picos_der[0]] < U_Derecha[2] and ventana_izq[picos_izq[0]] > U_Derecha[1] \
        and ventana_der[picos_der[-1]] > U_Derecha[0] and ventana_izq[picos_izq[-1]] < U_Derecha[3]:
        Mov = 'D'

    elif ventana_der[picos_der[0]] > U_Izquierda[1] and ventana_izq[picos_izq[0]] <  U_Izquierda[3] \
        and ventana_der[picos_der[-1]] < U_Izquierda[2] and ventana_izq[picos_izq[-1]] > U_Izquierda[0]:
        Mov = 'I'

    return Mov


#Reproducir video

def play_video(file_name):

    video = cv2.VideoCapture(file_name)

    if (video.isOpened() == False):
        print ('Error')

    while (video.isOpened()):
        ret, frame = video.read()

        if ret == True:
            frame = cv2.resize(frame, dsize=(920, 780), interpolation=cv2.INTER_AREA)
            cv2.imshow("video", frame)

            if cv2.waitKey(15) & 0xFF == ord('m'):
                break

        else:
            break

    video.release()
    cv2.destroyAllWindows()

# Generador de movimientos aleatorios
def mov_list(mode='train', **movimientos):
    if mode=='train':
        reps = 3
    elif mode=='test':
        reps = 1
    else:
        raise Exception('Invalid mode. Expected "train" or "test"')

    movs = []

    for mov in movimientos:

        movs = movs + ([movimientos[mov]] * reps)

    random.shuffle(movs)

    return movs

#Reproducir video
def play_vid(mov):

    path = 'mp4//' + mov + '.mp4'

    video = cv2.VideoCapture(path)

    scale = .2

    while video.isOpened():
        ret, frame = video.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        dsize = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        frame = cv2.resize(frame, dsize)
        cv2.imshow('frame', frame)
        if cv2.waitKey(80) == 80:
            break
    video.release()
    cv2.destroyAllWindows()





