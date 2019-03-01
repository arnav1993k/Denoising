#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:11:58 2019

@author: arnavk
"""
import os
import numpy as np
import io
import PIL
import torch
from torchvision.transforms import ToTensor
import librosa
from matplotlib import pyplot as plt
import soundfile as sf
import python_speech_features as psf
import math
def get_clean_file(noisy_file,clean_path):
    noisy_name = noisy_file.split("/")[-1]
    for c in clean_path:
        if c.split("/")[-1]==noisy_name:
            return c

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if fullPath[-3:]=="wav" or fullPath[-4:]=="flac":
                allFiles.append(fullPath)
                
    return allFiles
def normalize_signal(signal):
    return signal / np.max(np.abs(signal))


def plot_spectrogram(inputs,band=300,labels=None,save=False, fname="Figure"):
    if len(inputs)!=len(labels):
        print("Size of inputs and labels do not match")
        return
    fig, ax = plt.subplots(nrows=len(inputs),ncols=2, figsize=(len(inputs)*6,16),gridspec_kw={"width_ratios":[1, 0.01]})
    vmin = min(image.min() for image in inputs)
    vmax = max(image.max() for image in inputs)
    col_bar = [0 for i in inputs]
    for i in range(len(inputs)):
        cm=ax[i][0].imshow(inputs[i][:,:band], interpolation='nearest', cmap=plt.cm.afmhot, origin='lower', aspect='auto',vmin=vmin,vmax=vmax)
        ax[i][0].set_title('{} spectrogram'.format(labels[i]))
        col_bar[i]=plt.colorbar(cm,cax=ax[i][1])
        col_bar[i].set_label('{}'.format(labels[i]))
    plt.figtext(0.05, 0.01, fname, fontsize=8, va="bottom", ha="left")
    if save:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()
        return buf
    else:
        plt.show()
    plt.close()
def im_to_tensorboard(buf,writer,iteration):
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    writer.add_image('Image', image, iteration)

def custom_gla(initial_phase, magnitudes, n_fft, hop_length, n_iters):
    phase =initial_phase
    complex_spec = magnitudes* phase
    signal = librosa.istft(complex_spec, hop_length=hop_length)
    if not np.isfinite(signal).all():
        print("WARNING: audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec, hop_length=hop_length)
    return signal
def get_sound(specs,writer,ph,iteration,gla=False, psf=False):
    text = ["Original","Denoised","Target"]
    for i in range(len(specs)):
        if psf:
            # specs[i] *= 2
            mag = get_psf_mag(specs[i],512,64)
            phase = ph.T
        else:
            mag = get_mag(specs[i], 512,64).T
            phase = ph
        if gla:
            signal = custom_gla(ph, mag, 512, hop_length=160, n_iters=10)
        else:
            cmplx = mag.T  * phase  #f x t
            signal = librosa.istft(cmplx, hop_length=160)
            signal = normalize_signal(signal)
        signal = signal.reshape((1,-1))
        writer.add_audio(text[i], signal, iteration, sample_rate=16000)

def get_mel(signal, sample_freq, n_fft, n_window_size, n_window_stride, features, pad_to = 8):
    mel_basis = librosa.filters.mel(sr=sample_freq,
                                    n_fft=n_fft,
                                    n_mels=features,
                                    htk=True,
                                    norm=None,
                                    fmin=0,
                                    fmax=8000)
    mag, ph = librosa.magphase(librosa.stft(y=signal, n_fft=n_fft,hop_length=n_window_stride,win_length=n_window_size),power=1)
    features = np.dot(mel_basis, mag).T
    pad_value = 1e-2
    if pad_to > 0:
        num_pad = pad_to - ((len(features) + 1) % pad_to) + 1
        features = np.pad(
            features,
            ((0, num_pad), (0, 0)),
            "constant",
            constant_values=pad_value
        )
        ph =np.pad(
            ph,
            ((0, num_pad), (0, 0)),
            "constant",
            constant_values=0
        )
        assert features.shape[0] % pad_to == 0
    features = np.clip(features, a_min=1e-2, a_max=None)
    features = np.log(features)
    return features,ph
# def get_mel(signal, sample_freq, n_fft, size,stride, features, pad_to=8):
#     mel_basis = librosa.filters.mel(sr=16000,
#                                     n_fft=n_fft,
#                                     n_mels=features,
#                                     htk=True,
#                                     norm=None,
#                                     fmin=0,
#                                     fmax=8000)
#     mag, ph = librosa.magphase(librosa.stft(y=signal, n_fft=n_fft, win_length=size, hop_length=stride), power=1)
#     feat = np.dot(mel_basis, mag).T
#     if pad_to>0:
#         feat = np.pad(feat, [(0, pad_to - feat.shape[0] % pad_to), (0, 0)], mode='constant',
#                           constant_values=1e-2)
#     feat = np.clip(feat, a_min=1e-2, a_max=None)
#     return feat
def get_psf_mel(signal, sample_freq, n_fft, n_window_size, n_window_stride, features, pad_to = 8):
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(
        np.int16)

    audio_duration = len(signal) * 1.0 / sample_freq

    # making sure length of the audio is divisible by 8 (fp16 optimization)
    length = 1 + int(math.ceil(
        (1.0 * signal.shape[0] - n_window_size) / n_window_stride
    ))
    if pad_to > 0:
        if length % pad_to != 0:
            pad_size = (pad_to - length % pad_to) * n_window_stride
            signal = np.pad(signal, (0, pad_size), mode='constant')
    highfreq = sample_freq / 2
    signal = psf.sigproc.preemphasis(signal, 0.97)
    frames = psf.sigproc.framesig(signal, n_window_size, n_window_stride, np.hanning)
    complex_spec = np.fft.rfft(frames, n_fft)
    mag_spec= np.absolute(complex_spec)
    ph = np.exp(1.j * np.angle(complex_spec))
    pspec = 1.0 / n_fft * np.square(mag_spec)
    fb = psf.base.get_filterbanks(features, n_fft, sample_freq, 0, highfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat= np.log(feat)
    mean = np.mean(feat)
    std_dev = np.std(feat)
    feat = (feat - mean) / std_dev
    # feat /= 2
    return feat, ph

def run_test(model, input_path, target_path, features, max_length,device, psf=False):
    i = np.random.choice(len(input_path),1)[0]
    signal, sample_freq = sf.read(input_path[i])
    target, sample_freq = sf.read(get_clean_file(input_path[i],target_path))
    if not psf:
        X, ph = get_mel(signal,sample_freq, 512, 320, 160, 64,pad_to=0)
        y, _ = get_mel(target,sample_freq, 512, 320, 160, 64,pad_to=0)
    else:
        X, ph = get_psf_mel(signal, sample_freq, 512, 320, 160, 64, pad_to=0)
        y, _ = get_psf_mel(target, sample_freq, 512, 320, 160, 64, pad_to=0)
    x = np.expand_dims(X,axis=0)
    x = np.expand_dims(x,axis=0)
    x = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        dec = model(x)
        dec = dec.cpu().numpy()
        dec = dec.squeeze()
    specs = [X.T,dec.T,y.T]
    return specs,ph, input_path[i]

def run_test_apex(model, file_path, features, max_length,device, psf=False):
    i = np.random.choice(len(file_path),1)[0]
    signal, sample_freq = sf.read(file_path[i][0])
    target, sample_freq = sf.read(file_path[i][1])
    if not psf:
        X, ph = get_mel(signal,sample_freq, 512, 320, 160, 64,pad_to=0)
        y, _ = get_mel(target,sample_freq, 512, 320, 160, 64,pad_to=0)
    else:
        X, ph = get_psf_mel(signal, sample_freq, 512, 320, 160, 64, pad_to=0)
        y, _ = get_psf_mel(target, sample_freq, 512, 320, 160, 64, pad_to=0)
    x = np.expand_dims(X,axis=0)
    x = np.expand_dims(x,axis=0)
    x = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        dec = model(x)
        dec = dec.cpu().numpy()
        dec = dec.squeeze()
    specs = [X.T,dec.T,y.T]
    return specs,ph, file_path[i][0]

def get_mag(mel, n_fft, features):
    mel = np.exp(mel)
    mel_basis = librosa.filters.mel(sr=16000,
                                    n_fft=n_fft,
                                    n_mels=features,
                                    htk=True,
                                    norm=None,
                                    fmin=0,
                                    fmax=8000)
    mag = np.dot(mel_basis.T,mel)
    return mag

def get_psf_mag(mel, n_fft, features):
    mel = np.exp(mel)
    mel_basis = psf.base.get_filterbanks(features, n_fft, 16000, 0, 8000)
    mag = np.dot(mel.T,mel_basis)
    return mag