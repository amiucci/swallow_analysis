import numpy as np
import librosa
import librosa.display
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks, find_peaks_cwt
import scipy.signal as signal
import pywt
import sys
import noisereduce as nr
import soundfile as sf
#import spkit
#print('spkit-version ', spkit.__version__)
#import spkit as sp
#from spkit.cwt import ScalogramCWT
#from spkit.cwt import compare_cwt_example

LC = float(sys.argv[2])
HC = float(sys.argv[3])
FRAME_RATE = 22050

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y 

def bandpass_filter(buffer):
    return butter_bandpass_filter(buffer, LC, HC, FRAME_RATE)

def spectrogram(x, Fs, i, mel):
    N, H = 2048, 1024
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    if (mel == 'True' or mel == 'true'):
####### MEL SPECTROGRAM
        X_mag, _ = librosa.magphase(X)
        mel_scale_X = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=X_mag, sr=Fs), ref=np.max)
        print("mel")
        sg = librosa.display.specshow(mel_scale_X, y_axis='mel', x_axis='time', sr=Fs, hop_length=H, cmap='gnuplot2', ax=axs[i])
###### CONVENTIONAL SPECTROGRAM
    else:
        print("linear")
        Y = np.abs(X)
        sg = librosa.display.specshow(librosa.amplitude_to_db(Y, ref=np.max), y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gnuplot2', ax=axs[i])
    return sg


filepath, filename = str(sys.argv[1]).split('/',2)
filt_filename = 'filt_'+filename[:-4]+'_'+str(LC)+'-'+str(HC)+filename[-4:]
filtNR_filename = 'filtNR_'+filename[:-4]+'_'+str(LC)+'-'+str(HC)+filename[-4:]


samplerate, data = wavfile.read(filepath+'/'+filename)
filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')
wavfile.write(filepath+'/'+filt_filename, samplerate, filtered)

data2 , samplerate2 =sf.read(filepath+'/'+filt_filename)
print(len(data2), samplerate2)
noise , samplerate3 = sf.read(filepath+'/noise_16bit.wav')
filtered_NR = nr.reduce_noise(audio_clip=librosa.to_mono(data2), noise_clip=noise)
#filtered_NR = filtered_NR * 10
wavfile.write(filepath+'/'+filtNR_filename, samplerate2, filtered_NR)


x_noise, Fs_noise = librosa.load(filepath+'/noise_16bit.wav', sr = FRAME_RATE)
x_data, Fs_data = librosa.load(filepath+'/'+filename, sr = FRAME_RATE)
x_dataF, Fs_dataF = librosa.load(filepath+'/'+filt_filename, sr = FRAME_RATE)
x_dataFNR, Fs_dataFNR = librosa.load(filepath+'/'+filtNR_filename, sr = FRAME_RATE)

x_array = [x_noise, x_data, x_dataF, x_dataFNR]
x_title = [filepath+'/noise.wav', filepath+'/'+filename, filepath+'/'+filt_filename, filepath+'/'+filtNR_filename]
x_fs = [Fs_noise, Fs_data, Fs_dataF, Fs_dataFNR]
counter = 0
#plt.figure(figsize=(8, 6))
fig, axs = plt.subplots(4,1,figsize=(8, 7))
for x in x_array:
    img = spectrogram(x, x_fs[counter], counter, sys.argv[4])
    plt.colorbar(img, ax=axs[counter], format='%+2.0f dB')
    axs[counter].set_title(x_title[counter])
    if not (sys.argv[4] == 'True' or sys.argv[4] == 'true' ):
        axs[counter].set_yscale("log")
    axs[counter].set_ylim([10,20000])
    axs[counter].set_xlabel('Time [s]')
    axs[counter].set_ylabel('Frequency [Hz]')
    counter = counter+1
#print("Sampling rate is: ", sr)
#plt.figure()
#plt.show()
fig.tight_layout()
#plt.show()

d1 = filtered_NR
dmax = abs(d1).max()
dd = d1[252500:307750]
plot_scaling = 0.3#0.05

fig2, axs2= plt.subplots(3,1,figsize=(8,6))
widths = np.arange(2,50,0.1)#,np.arange(1001,2001,100))
#widths_mh = np.append(np.arange(7,31,0.1),np.arange(31,1001, 10))#,np.arange(1001,2001,100))
#print(widths, pywt.scale2frequency("morl",(widths**2)/samplerate2))
t=np.linspace(0,dd.size/samplerate2,dd.size)
cwtmatr,freq = pywt.cwt(dd, widths**2, "morl", sampling_period=(1/samplerate2))
print(freq)
#cwtmatr_mh,freq_mh = pywt.cwt(dd, widths_mh, "gaus4", sampling_period=(1/samplerate2))
    

print(t.size,dd.size)
line0 = axs2[0].plot(t, dd)
plt.setp(line0, linewidth=0.5)
axs2[0].set_xlim(0,t[t.size-1])
axs2[0].set_ylim(-dmax*plot_scaling,dmax*plot_scaling)
axs2[0].set_xlabel('Time [s]')
axs2[0].set_ylabel('Amplitude [arb.un.]')
#axs2[1].imshow(cwtmatr, cmap='PiYG', extent = [t[0],t[t.size-1],freq[0],freq[freq.size-1]])
axs2[1].pcolormesh(t,freq,cwtmatr, cmap='PiYG', vmax=dmax*plot_scaling, vmin=-dmax*plot_scaling)
axs2[1].set_yscale("log")
axs2[1].set_xlabel('Time [s]')
axs2[1].set_ylabel('Frequency [Hz]')
axs2[2].pcolormesh(t,freq,librosa.amplitude_to_db(cwtmatr, ref=np.max) , cmap='gnuplot2')
axs2[2].set_yscale("log")
axs2[2].set_xlabel('Time [s]')
axs2[2].set_ylabel('Frequency [Hz]')


fig2.tight_layout()
#dd2 = d1[-30000:]
#fig3, axs3= plt.subplots(3,1,figsize=(8,6))
#t2=np.arange(0,dd2.size)/samplerate2
#cwtmatr2,freq2 = pywt.cwt(dd2, widths, "morl", sampling_period=(1/samplerate2))
#cwtmatr2_mh,freq2_mh = pywt.cwt(dd2, widths_mh, "gaus4", sampling_period=(1/samplerate2))
#line1 = axs3[0].plot(t2,dd2)
#plt.setp(line1, linewidth=0.5)
#axs3[0].set_xlim(0,t2[t2.size-1])
#axs3[0].set_ylim(-dmax*plot_scaling,dmax*plot_scaling)
#axs3[0].set_xlabel('Time [s]')
#axs3[0].set_ylabel('Amplitude [arb.un.]')
#axs3[1].imshow(cwtmatr2, cmap='PiYG', extent = [t2[0],t2[t.size-1],freq2[0],freq2[freq2.size-1]])
#axs3[1].pcolormesh(t2,freq2,cwtmatr2, cmap='PiYG', vmax=dmax*plot_scaling, vmin=-dmax*plot_scaling)
#axs3[1].set_yscale("log")
#axs3[1].set_xlabel('Time [s]')
#axs3[1].set_ylabel('Frequency [Hz]')
#axs3[2].pcolormesh(t2,freq2,(np.abs(cwtmatr2_mh))**2, cmap='gnuplot2')
#axs3[2].set_yscale("log")
#axs3[2].set_xlabel('Time [s]')
#axs3[2].set_ylabel('Frequency [Hz]')
#fig3.tight_layout()
#fig3, axs3= plt.subplots(2,1,figsize=(8,4))
#cwtmatr_db,freq_db = pywt.cwt(librosa.amplitude_to_db(dd), widths, "mexh", sampling_period=(1/samplerate2))
#line1 = axs3[0].plot(librosa.amplitude_to_db(dd, ref=np.min))
#plt.setp(line1, linewidth=0.5)
#axs3[0].set_xlim(0,len(dd))
#axs3[1].pcolor(t,freq_db,cwtmatr_db, cmap='bone', vmax=abs(cwtmatr_db).max(), vmin=0)

plt.show()


