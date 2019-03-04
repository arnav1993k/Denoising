#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:34:35 2019

@author: arnavk
"""


import numpy as np
import torch
from utils import *
from autoencoder import AutoEncoder_Lib
import librosa
import soundfile as sf
params={
       "data_specs":{
           "noisy_path":"/raid/Speech/Natural Noise/sounds/village",
           "clean_path":"/raid/Speech/LibriSpeech/dev-other-wav-16KHz",
           "filemap_path":"/raid/Speech/"
            },
        "spectrogram_specs":{
            "window_size":20e-3,
            "window_stride":10e-3,
            "type":"logfbank",
            "features":64
        },
        "training":{
            "save_path":"/raid/Speech/test_models/model_lib_full_apx.ckpt",
            "train_test_split":1,
            "batch_size":64,
            "num_epochs":4000,
            "device":"cuda:3",
            "seq_length":400 #keep odd
        }
       }

features = params["spectrogram_specs"]["features"]

#train config
device = torch.device(params["training"]["device"])
max_length=params["training"]["seq_length"]
batch_size = params["training"]["batch_size"]
num_epochs = params["training"]["num_epochs"]
save_path = params["training"]["save_path"]
split = params["training"]["train_test_split"]

#data config
noise_path = sorted(getListOfFiles(params["data_specs"]["noisy_path"]))
clean_path = sorted(getListOfFiles(params["data_specs"]["clean_path"]))
split_point = int(split*len(clean_path))

actual_path_train = clean_path[:split_point]
clean_path_train = clean_path[:split_point]
actual_path_test = clean_path[split_point:]
clean_path_test = clean_path[split_point:]

#spectrogram config
features_type=params["spectrogram_specs"]["type"]
window_size=params["spectrogram_specs"]["window_size"]
window_stride=params["spectrogram_specs"]["window_stride"]

autoencoder = AutoEncoder_Lib([1,32,128,32,1]).to(device)
autoencoder.load_state_dict(torch.load(save_path))
all_noise = []
for n in noise_path:
    noise,sr = librosa.load(n, sr=16000)
    all_noise.append(noise)

def test(model,base_path):
    # clean_dir = base_path + "/clean/"
    # if not os.path.isdir(base_path):
    #     os.mkdir(base_path)
    #     os.mkdir(clean_dir)
    # for f in actual_path_train:
    #     target, sample_freq = sf.read(f)
    #     duration = int(len(target) / sample_freq)
    #     name = f.split("/")[-1]
    #     n_window_size = int(sample_freq * window_size)
    #     n_window_stride = int(sample_freq * window_stride)
    #     y, _ = get_mel(target, sample_freq, 512, n_window_size, n_window_stride, features, pad_to=8)
    #     np.savez(clean_dir + name + '.npz', features=y, duration=duration)

    for p in ["no_noise"]:
        store_path = base_path+"/noisy_{}db".format(p)
        denoised_dir = store_path + "/denoised/"
        noisy_dir = store_path + "/noisy/"
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
            os.mkdir(noisy_dir)
            os.mkdir(denoised_dir)
        for f in actual_path_train:
            target, sample_freq = sf.read(f)
            duration = int(len(target) / sample_freq)
            name = f.split("/")[-1]
            # percent = np.random.randint(50, 100, 1)[0]
            # signal, target = get_noisy(target, noise, p, percent)
            n_window_size = int(sample_freq * window_size)
            n_window_stride = int(sample_freq * window_stride)
            X, _ = get_mel(target, sample_freq, 512, n_window_size, n_window_stride, features, pad_to=8)
            x = np.expand_dims(X, axis=0).astype(np.float32)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            x = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                dec = model(x)
            dec = dec.cpu().numpy().squeeze()
            np.savez(denoised_dir + name + '.npz', features=dec, duration=duration)
            np.savez(noisy_dir + name + '.npz', features=X, duration=duration)
test(autoencoder,"/raid/Speech/ae_outputs")