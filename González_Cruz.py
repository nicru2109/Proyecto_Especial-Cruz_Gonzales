##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import scipy.signal as sig


##
def f_FFTfilt(v_sig, s_SRate, str_type, v_cutFreq):

    fft = np.fft.fft(v_sig)
    fft_freq = np.fft.fftfreq(len(v_sig), 1 / s_SRate)

    ini = v_cutFreq[0]
    fin = v_cutFreq[1]

    i = 0

    ind1 = 0
    ind2 = 0

    while 1:

        if fft_freq[i] < ini:
            ind1 = i
        if fft_freq[i] > fin:
            ind2 = i
            break
        i += 1

    if str_type == 'Cut':
        fft[ind1:ind2] = fft[ind1:ind2] * 1E-100
    elif str_type == 'Pass':
        fft[0:ind1] = fft[0:ind1] * 1E-100
        fft[ind2:] = fft[ind2:] * 1E-100
    else:
        print('Instrucción inválida')

    v_filtSig = np.fft.irfft(fft, len(v_sig))

    return v_filtSig

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

def f_Int(v_sig,s_SRate,s_Win):
    s_winSam = s_Win * s_SRate
    v_Filt = np.zeros(len(v_sig))
    for idx in range(len(v_sig)):
        s_ini = int(idx - s_winSam / 2)
        s_fin = int(idx + s_winSam / 2)
        if s_ini < 0:
            s_ini = 0
        elif s_fin > len(v_sig):
            s_fin = len(v_sig)
        v_temp = v_sig[s_ini:s_fin]
        v_Filt[idx] = np.sum(v_temp)
    return v_Filt

def f_FindRPeaks(v_sig,s_SRate, v_cutFreq):

    v_Filt = f_FFTfilt(v_sig, s_SRate, 'Cut', v_cutFreq)
    v_Diff = np.zeros(np.size(v_Filt))
    v_Diff[1:]=np.diff(v_Filt)
    v_Diff[0] = v_Filt[0]
    v_Cuad = v_Diff**2
    v_Int = f_Int(v_Cuad,s_SRate,0.1)
    st_locs = sig.find_peaks(v_Int,prominence= 500)
    v_locs = st_locs[0]
    s_win = 0.5*s_SRate
    for idx in range(len(v_locs)):
        s_ini = int(v_locs[idx] - s_win)
        s_fin = int(v_locs[idx] + s_win)
        if s_ini < 0:
            s_ini = 0
        elif s_fin > len(v_Filt):
            s_fin = len(v_Filt)
        v_temp = v_Filt[s_ini:s_fin]
        st_TemLocs = sig.find_peaks(v_temp)
        v_tLocs = st_TemLocs[0]
        v_H = v_temp[v_tLocs]
        s_max = np.argmax(v_H)
        v_locs[idx] = s_ini + v_tLocs[s_max]
    return np.unique(v_locs)

def f_taco(v_locs, s_SRate):

    taco = np.diff(v_locs)
    taco = taco * (1/s_SRate)

    desv = np.std(taco)
    mean = np.mean(taco)

    bin = np.abs(taco - mean) > 3 * desv
    taco = np.delete(taco, bin)

    return taco

def f_vTime(s_SRate,s_len,b_cero):
    v_time = np.linspace(0,s_len/s_SRate,s_len)
    if b_cero:
        v_time = v_time - (np.max(v_time)/2)
    return v_time


def f_TimeFreq(v_Sig, S_sRate, V_freq, S_win):

    wind = int((S_win * S_sRate) / 2)
    temp = np.fft.rfftfreq(int(2 * wind + 1), 1 / S_sRate)
    ini = V_freq[0]
    fin = V_freq[1]
    
    i = 0

    ind1 = 0
    ind2 = temp[-1]

    while 1:
        if temp[i] < ini:
            ind1 = i
        if temp[i] > fin:
            ind2 = i
            break
        i += 1

    Mtf = np.zeros((ind2 - ind1, int(len(v_Sig) - (2 * wind))))

    i = wind

    while i in range(len(v_Sig) - wind):

        ini = int(np.round(i - wind))
        fin = int(np.round(i + wind))

        temp = v_Sig[ini:fin]

        fft = np.abs(np.fft.rfft(temp))
        freqs = np.fft.rfftfreq(len(temp), 1 / S_sRate)
        frecs = freqs[ind1:ind2]

        Mtf[:, i - wind - 1]  = fft[ind1:ind2].T

        i +=1

    T = f_vTime(S_sRate, Mtf.shape[1], True)

    # plt.figure()
    plt.imshow(Mtf, cmap='hot', aspect='auto', extent = [T[0], T[-1], V_freq[1], V_freq[0]])
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar()

    return Mtf


def f_GenerateMeanGraph(data, w, t):
    ind = int(w * t)

    a = data.shape[1]
    b = data[:, int((np.ceil(a/2) - 1 - int(ind/2))):int((np.ceil(a/2) + int(ind/2)))]


    c = np.sum(b, 0)/b.shape[0]

    return c

def f_cut(v_Sig, v_locs, s_SRate, s_Win):

    win = np.round(s_SRate * s_Win)
    m_data = np.zeros((len(v_locs), int((win * 2) + 1)))

    for i in range(len(v_locs)):

        ini = int(v_locs[i] - win)
        fin = int(v_locs[i] + win + 1)

        a = 0
        b = int((win * 2) + 1)

        if ini < 0:
            a = - ini
            ini = 0
        if fin > len(v_Sig):
            b = b - (fin - len(v_Sig))
            fin = len(v_Sig)

        temp = v_Sig[ini:fin]


        m_data[i, a:b] = temp

    return m_data


## carga de los datos


dfActividad = pd.read_csv(os.path.join('DatosMiercoles1', 'OpenBCI-RAW-2021-04-14_07-20-49.txt'), header=4)
dfDescanso = pd.read_csv(os.path.join('DatosMiercoles1', 'OpenBCI-RAW-2021-04-14_07-22-16.txt'), header=4)
dfReposo = pd.read_csv(os.path.join('DatosMiercoles1', 'OpenBCI-RAW-2021-04-14_07-17-18.txt'), header = 4)

rep = dfReposo[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2',' EXG Channel 6']].to_numpy()
act = dfActividad[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2',' EXG Channel 6']].to_numpy()
desc = dfDescanso[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2',' EXG Channel 6']].to_numpy()

S_sRate = 250

##Prueba de filtrado
v_EEG_prueba = rep[200:, 3]*-1
v_EEG_prueba = sig.detrend(v_EEG_prueba)

t_0 = time.time()
v_EEG_Filt_FFt = f_FFTfilt(v_EEG_prueba, S_sRate, 'Cut', [50, 70])
t_1 = time.time() - t_0
print(t_1)

t_0_Av = time.time()
v_EEG_Filt_Av = f_AvFlt(v_EEG_prueba, S_sRate, 0.05)
t_1_Av = time.time() - t_0
print(t_1_Av)

plt.figure()
plt.plot(v_EEG_prueba)
plt.show()

plt.figure()
plt.plot(v_EEG_Filt_FFt)
plt.show()
## Find peaks
v_ECG_rep = rep[200:7200, 3]* -1
v_ECG_rep = sig.detrend(v_ECG_rep)

v_ECG_act = act[200:7200, 3]* -1
v_ECG_act = sig.detrend(v_ECG_act)

v_ECG_desc = desc[200:7200, 3] * -1
v_ECG_desc = sig.detrend(v_ECG_desc)

v_ECG_Filt_rep = f_FFTfilt(v_ECG_rep, S_sRate, 'Pass', [0.1, 30])
v_ECG_Filt_act = f_FFTfilt(v_ECG_act, S_sRate, 'Pass', [0.1, 30])
v_ECG_Filt_desc = f_FFTfilt(v_ECG_desc, S_sRate, 'Pass', [0.1, 30])

ECG_RPeaks_rep = f_FindRPeaks(v_ECG_rep, S_sRate, [50, 70])
ECG_RPeaks_act = f_FindRPeaks(v_ECG_act, S_sRate, [50, 70])
ECG_RPeaks_desc = f_FindRPeaks(v_ECG_desc, S_sRate, [50, 70])


fig, axs = plt.subplots(3,1)
#axs.set_subtittle('Picos R en ECG')
axs[0].set_title('Reposo')
axs[0].set_ylabel('Amplitud [v]')
axs[0].plot(v_ECG_Filt_rep)
axs[0].plot(ECG_RPeaks_rep, v_ECG_Filt_rep[ECG_RPeaks_rep], 'o')

axs[1].set_title('Actividad')
axs[1].set_ylabel('Amplitud [v]')
axs[1].plot(v_ECG_Filt_act)
axs[1].plot(ECG_RPeaks_act, v_ECG_Filt_act[ECG_RPeaks_act], 'o')

axs[2].set_title('Descanso')
axs[2].set_xlabel('No. Muestra')
axs[2].set_ylabel('Amplitud [v]')
axs[2].plot(v_ECG_Filt_desc)
axs[2].plot(ECG_RPeaks_desc, v_ECG_Filt_desc[ECG_RPeaks_desc], 'o')



## vector promedio
v_EEG_rep0 = rep[200:, 0]
v_EEG_rep1 = rep[200:, 1]
v_EEG_rep2 = rep[200:, 2]

v_EEG_act0 = act[200:, 0]
v_EEG_act1 = act[200:, 1]
v_EEG_act2 = act[200:, 2]

v_EEG_desc0 = desc[200:, 0]
v_EEG_desc1 = desc[200:, 1]
v_EEG_desc2 = desc[200:, 2]

signals_rep = [v_EEG_rep0, v_EEG_rep1, v_EEG_rep2]
signals_act = [v_EEG_act0, v_EEG_act1, v_EEG_act2]
signals_desc = [v_EEG_desc0, v_EEG_desc1, v_EEG_desc2]

V_Mean = []

for i in range (len(signals_rep)):
    signals_rep[i] = sig.detrend(signals_rep[i])
    Filt = f_FFTfilt(signals_rep[i], S_sRate, 'Cut', [50, 70])
    Cut = f_cut(Filt, ECG_RPeaks_rep, S_sRate, 2)
    V_Mean.append(f_GenerateMeanGraph(Cut, S_sRate, 2))

for i in range (len(signals_act)):
    signals_act[i] = sig.detrend(signals_act[i])
    Filt = f_FFTfilt(signals_act[i], S_sRate, 'Cut', [50, 70])
    Cut = f_cut(Filt, ECG_RPeaks_act, S_sRate, 2)
    V_Mean.append(f_GenerateMeanGraph(Cut, S_sRate, 2))

for i in range (len(signals_desc)):
    signals_desc[i] = sig.detrend(signals_desc[i])
    Filt = f_FFTfilt(signals_desc[i], S_sRate, 'Cut', [50, 70])
    Cut = f_cut(Filt, ECG_RPeaks_desc, S_sRate, 2)
    V_Mean.append(f_GenerateMeanGraph(Cut, S_sRate, 2))

##Función tiempo frecuencia
# EEG en reposo
plt.figure()
plt.subplot(3, 1, 1)
plt.title('Análisis Tiempo Frecuencia (EEG Reposo) \n Mapa de calor (Canales 0, 1 y 2)')
f_TimeFreq(V_Mean[0], S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 2)
f_TimeFreq(V_Mean[1], S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 3)
f_TimeFreq(V_Mean[2], S_sRate, [0.1,40], 0.2)

# EEG en actividad
plt.figure()
plt.subplot(3, 1, 1)
plt.title('Análisis Tiempo Frecuencia (EEG Actividad) \n Mapa de calor (Canales 0, 1 y 2)')
f_TimeFreq(V_Mean[3], S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 2)
f_TimeFreq(V_Mean[4], S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 3)
f_TimeFreq(V_Mean[5], S_sRate, [0.1,40], 0.2)

#EEG Descanso
plt.figure()
plt.subplot(3, 1, 1)
plt.title('Análisis Tiempo Frecuencia (EEG Descanso) \n Mapa de calor (Canales 0, 1 y 2)')
f_TimeFreq(V_Mean[6], S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 2)
f_TimeFreq(V_Mean[7], S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 3)
f_TimeFreq(V_Mean[8], S_sRate, [0.1,40], 0.2)

# ECG
cut_ECG_rep = f_cut(v_ECG_Filt_rep, ECG_RPeaks_rep, S_sRate, 2)
v_ECGmean_rep = f_GenerateMeanGraph(cut_ECG_rep, S_sRate, 2)

cut_ECG_act = f_cut(v_ECG_Filt_act, ECG_RPeaks_act, S_sRate, 2)
v_ECGmean_act = f_GenerateMeanGraph(cut_ECG_act, S_sRate, 2)

cut_ECG_desc = f_cut(v_ECG_Filt_desc, ECG_RPeaks_desc, S_sRate, 2)
v_ECGmean_desc = f_GenerateMeanGraph(cut_ECG_desc, S_sRate, 2)

plt.figure()
plt.subplot(3, 1, 1)
plt.title('Análisis Tiempo Frecuencia de ECG \n Mapa de calor')
f_TimeFreq(v_ECGmean_rep, S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 2)
f_TimeFreq(v_ECGmean_act, S_sRate, [0.1,40], 0.2)
plt.subplot(3, 1, 3)
f_TimeFreq(v_ECGmean_desc, S_sRate, [0.1,40], 0.2)

##Tacograma
#Reposo

tac_rep = f_taco(ECG_RPeaks_rep, S_sRate)
tac_act = f_taco(ECG_RPeaks_act, S_sRate)
tac_desc = f_taco(ECG_RPeaks_desc, S_sRate)

T1 = f_vTime(S_sRate, len(v_EEG_rep0), False)
Trep = T1[ECG_RPeaks_rep]
Tact = T1[ECG_RPeaks_act]
Tdesc = T1[ECG_RPeaks_desc]

Trep = Trep[1:]
Tact = Tact[1:]
Tdesc = Tdesc[1:]

plt.figure()

plt.subplot(3, 1, 1)
plt.title('Tacrogama ECG Repaso')
plt.plot(Trep, tac_rep)


plt.subplot(3, 1, 2)
plt.title('Tacrogama ECG Actividad')
plt.ylabel('')
plt.plot(Tact, tac_act)

plt.subplot(3, 1, 3)
plt.title('Tacrogama ECG Descanso')
plt.plot(Tdesc, tac_desc)
