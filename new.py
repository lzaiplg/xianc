import time

import torch
import os
import librosa
import numpy as np
from ConvtasnetModels import ConvTasNet
from scipy import signal as sig
from pit_criterion import cal_loss
import soundfile
from copy import deepcopy
import scipy
import resampy
from numba import njit
from scipy.fftpack import fft, ifft
from scipy.special import expn
import scipy.special as special
from sympy import *
from scipy.signal import butter, lfilter, freqz
from torch.utils.data import Dataset,DataLoader
from Preprocess import mix_speechwav_noisewav
from MyDataSet import signal_pad
from pydub import AudioSegment
from scipy import signal as sig
from scipy.fftpack import fft, ifft,fftshift
import soundfile as sf
import test_mmse_1 as mmse
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram
import pywt
import struct
import socket
import sounddevice as sd
from queue import Queue
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2 import QtWidgets
# from PySide2.QtCore import QFile, QTimer
from PyQt5.QtCore import QTimer
import threading
from socket import *
from openal import *
import pyogg
class Digital_Preprocess:
    def __init__(self):
        self.path = 'denosie_audio'
        self.infile1 = 'ACTS2235-0_short.txt'
        self.filereads = [self.infile1]
        self.wavtype = '.wav'
        self.filetype = 'txt'
        self.filetype2 = 'bin'
        self.outputpath = "Experiment/denoised_wavs/den_SSB001100111112.wav"
        self.outputpath_neg = "Experiment/denoised_wavs/den_SSB001100111112_neg.wav"
        self.outputpath_original = "Experiment/denoised_wavs/den_SSB001100111112_orig.wav"
        self.outputpath_original_neg = "Experiment/denoised_wavs/den_SSB001100111112_orig_neg.wav"
        self.complite_compensation = True
        self.carrir_wave_frequence = 50000
        self.target_samplerate = 22050
        self.original_ad_samplerate = 500000
        self.dt = 1/self.original_ad_samplerate
        self.voice_low = 100
        self.voice_high = 7000
        self.carrir_wave_bandpass = np.array([49.750e3, 50.250e3])
    def change_to_float(self,strp):
        temp = strp.split("E")
        num = float(temp[0])
        return num
    def load_list(self,f):
        infile = f
        self.write_file_name = infile
        data = pd.read_csv(infile, encoding='gbk',sep = ' ', skiprows=1, header=None,  dtype= np.float64)#chunksize=1000000,usecols=[1, 3],
        self.time_list = data[data.columns].values[:, 1:2]
        self.time_list = self.time_list.astype(float)
        self.data = data[data.columns].values[:, -1]
        self.data = self.data.astype(float)
        self.time_list_cauculate = self.time_list[1:5]
        time1 = self.time_list_cauculate[1]- self.time_list_cauculate[0]
        time2 = self.time_list_cauculate[3]- self.time_list_cauculate[2]
        if time1 == time2:
            self.dt = time2/1000
        else:
            self.dt = "time_not_equal!!!"
            exit()
        self.data = np.array(self.data/ 1).reshape(1,-1)[0]
        self.time_list = np.array(self.time_list/1000).reshape(1,-1)[0]
        self.fs = 1 / self.dt
        self.Center_Frequency = 50000
        return self.data
    def arc(self,data):
        f_data1 = data
        self.fs = 1 / self.dt
        self.Center_Frequency = 50000
        self.time_list = np.arange(0, len(f_data1) / self.fs, self.dt)
        if len(self.time_list) != len(f_data1):
            if len(self.time_list) > len(f_data1):
                self.time_list = self.time_list[0:-1]
            else:
                f_data1 = f_data1[0:-1]
        sin_carrier_wave = np.sin(2 * np.pi * self.carrir_wave_frequence *self.time_list)
        Fsin = f_data1 * sin_carrier_wave
        cos_carrier_wave = np.cos(2 * np.pi * self.carrir_wave_frequence * self.time_list)
        Fcos = f_data1 * cos_carrier_wave
        sr = 50000
        filted_data_SIN = Digital_Preprocess.butter_lowpass_filter(Fsin, self.voice_high, self.fs, 5)
        filted_data_COS = Digital_Preprocess.butter_lowpass_filter(Fcos, self.voice_high, self.fs, 5)
        warped_data = np.arctan2(filted_data_COS, -filted_data_SIN)
        # unwarped_data = np.unwrap(arctan)
        a = 1
        def unwrap(phase):
            k = 0
            L = len(phase)
            unwrappedPhase = phase
            diff = phase[1:-1] - phase[0: -2]
            for i in range(len(phase) - 2):
                unwrappedPhase[i] = phase[i] + 2 * np.pi * k
                if diff[i] >= np.pi:
                    k = k - 1
                elif diff[i] < -np.pi:
                    k = k + 1
            unwrappedPhase[-1] = phase[-1] + (2 * np.pi * k)
            return unwrappedPhase
        unwarped_data = unwrap(warped_data)
        resampled_data = resampy.resample(unwarped_data, self.original_ad_samplerate, sr)
        self.out_tmp = Digital_Preprocess.butter_highpass_filter(resampled_data, self.voice_low, sr, 5)
        tag_1 = '原始语音'
        return self.out_tmp, sr,tag_1
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        [b, a] = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = Digital_Preprocess.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y  # Filter requirements.
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        [b, a] = butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a
    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = Digital_Preprocess.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y  # Filter requirements.
    def butter_bandpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        [b, a] = butter(order, normal_cutoff, btype='bandpass', analog=False)
        return b, a
    def butter_bandpass_filter(data, cutoff, fs, order=5):
        b, a = Digital_Preprocess.butter_bandpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y  # Filter requirements.
class MainWindow():
    def __init__(self):
        super(MainWindow, self).__init__()
        self.q_mag_x = Queue(maxsize=0)
        self.q_mag_x_voice = Queue(maxsize=0)
        self.curve_num = 0
        self.historyLength = 160
        self.replay_buffer = []
        self.replay_audio_buffer = []
        self.DIG = Digital_Preprocess()
        self.ad_sampling_rate = self.DIG.original_ad_samplerate
        self.target_sampling_rate = self.DIG.target_samplerate
        self.DIG.load_list(mix_wav_path)
        self.BeginPleasear()
        # self.OpenSerial()
        self.i = 0
    def UpdateD(self):
        while True:
            self.replay_buffer = []
            while len(self.replay_buffer) < self.ad_sampling_rate:
                try:
                    mlist = self.q_mag_x.get()
                    self.replay_buffer.extend(mlist)
                except:
                    pass
            denoised_wav_list, SR, tag_1 = self.DIG.arc(self.replay_buffer)
            self.q_mag_x_voice.put(denoised_wav_list)
            print('TXT的数据get()成功！')
    def BeginPleasear(self):
        th1 = threading.Thread(target=self.Serial)
        th1.setDaemon(True)
        th1.start()
        time.sleep(1)
        th2 = threading.Thread(target=self.UpdateD)
        th2.start()
        th3 = threading.Thread(target=self.play_voice)
        th3.start()
    def play_voice(self):
        while True:
            sampling_rate = 50000
            play_delay = 2
            self.replay_audio_buffer = []
            while len(self.replay_audio_buffer) < self.target_sampling_rate*play_delay:
                try:
                    audio = self.q_mag_x_voice.get()
                    self.replay_audio_buffer.extend(audio)
                except:
                    pass
            time_sleep = len(self.replay_audio_buffer) / sampling_rate
            sd.play(self.replay_audio_buffer, sampling_rate)
            sd.wait()
            print('play')
    def Serial(self):
        txt = True
        fs = 2048
        if txt:
            while True:
                for i in range(0,len(self.DIG.data),fs):
                    ilist = self.DIG.data[i:i+fs]
                    self.q_mag_x.put(ilist)
                print('TXT数据载入')
                time.sleep(20)
        else:
            while (True):
                print('###########')
if __name__ == '__main__':
    modelpath = "Experiment/model_save/027000_1010_chang.pth"  ## 模型保存的 路径
    modelpath2 = "Experiment/model_save/077000.pth"  ## 模型保存的 路径
    mix_wav_path = r"denosie_audio_batch/" #要降噪的语音的路径
    clean_wav = r"denosie_audio_batch/ACTS2239.wav" #
    nosie_wav = r"denosie_audio_batch/ACTS2210.wav"  #
    outputpath = "Experiment/denoised_wavs2/"  ## 降噪后的 语音的保存位置。
    Dig = Digital_Preprocess()
    sample_rate = Dig.target_samplerate
    original_sr =  Dig.target_samplerate
    train_time = 3
    snr_list = [-50]
    nosie_flag = False
    denosie_dnn_flag = False
    generate_wavs_from_data =  False
    mmse_flag = False
    dsp_filter_flag =False
    a_flag = False
    wavelet_flag = False
    sub_space_flag = False
    file_flag = True
    logmmse_flag = False
    comp_flag = False
    pad_len = int(train_time * sample_rate)
    for rootdir, subdirs, files in os.walk(mix_wav_path):
        for f in files:
            mix_wav_path = os.path.join(rootdir, f)
            if file_flag:
                f_name = f.split('.')
                if f_name[-1] != Dig.filetype:
                    denoised_wav_list = 'error'
                    continue
                else:
                    app = QtWidgets.QApplication()
                    main = MainWindow()
                    denoised_wav_list = 1
            else:
                mix_wav, _ = librosa.load(mix_wav_path, sr=sample_rate)
                denoised_wav_list = mix_wav
    pass

