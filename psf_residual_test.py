#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:34:35 2019

@author: arnavk
"""


import numpy as np
import torch
import copy
import math
import torch
import torch.nn as nn
from utils import *
import pickle
from autoencoder import AutoEncoder
import librosa
import soundfile as sf
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from patter.model import ModelFactory
from patter.data import AudioDataset, BucketingSampler, audio_seq_collate_fn
from patter.decoder import GreedyCTCDecoder
from patter.data.features import LogFilterbankFeaturizer
from patter.evaluator import validate
from patter.models import SpeechModel

# writer = SummaryWriter('train_summary/residual/')
params={
       "data_specs":{
           "noisy_path":"/raid/Speech/LibriSpeech/clean_noisy_noise/test_batch_1/input/",
# "noisy_path":"/raid/Speech/LibriSpeech/dev-other-wav-16KHz/",
           "clean_path":"/raid/Speech/LibriSpeech/clean_noise/clean/",
            },
        "spectrogram_specs":{
            "window_size":20e-3,
            "window_stride":10e-3,
            "type":"logfbank",
            "features":64
        },
        "training":{
            "save_path":"/home/arnavk/Projects/AutoEncoder/model_noise_to_clear_image_simple_mel_v1.ckpt",
            "train_test_split":1,
            "batch_size":64,
            "num_epochs":4000,
            "device":"cuda:1",
            "seq_length":301 #keep odd
        }
       }
# seed_model_path = "checkpoints/torch_model.pt"
# seed_model = ModelFactory.load(seed_model_path)
# first_layer = seed_model
#del seed_model
#model config
# file_map = pickle.load(open("file_map.pkl","rb"))
#wts = np.load("first_layer.npz")
#first_layer = nn.Conv1d(64,256,11,2)
#first_layer.weight.data = torch.Tensor(wts["arr_0"]) 
features = params["spectrogram_specs"]["features"]

#train config
device = torch.device(params["training"]["device"])
max_length=params["training"]["seq_length"]
batch_size = params["training"]["batch_size"]
num_epochs = params["training"]["num_epochs"]
save_path = params["training"]["save_path"]
split = params["training"]["train_test_split"]

#data config
actual_path = sorted(getListOfFiles(params["data_specs"]["noisy_path"]))
clean_path = sorted(getListOfFiles(params["data_specs"]["clean_path"]))
split_point = int(split*len(actual_path))

actual_path_train = actual_path[:split_point]
clean_path_train = clean_path[:split_point]
actual_path_test = actual_path[split_point:]
clean_path_test = clean_path[split_point:]

#spectrogram config
features_type=params["spectrogram_specs"]["type"]
window_size=params["spectrogram_specs"]["window_size"]
window_stride=params["spectrogram_specs"]["window_stride"]

# autoencoder = AutoEncoder([1,32,64,32,1]).to(device)
# # first_layer = first_layer.to(device)
# loss_func = nn.KLDivLoss(reduction="batchmean")
# abs_loss = nn.MSELoss()
# dummy_input = torch.ones((batch_size,1,features,max_length)).to(device)
# writer.add_graph(autoencoder,dummy_input)
# autoencoder.load_state_dict(torch.load(save_path))

# def get_batch(batch_size, input_path, target_path, features, max_length=3000,features_type='logfbank',window_size=20e-3,window_stride=5e-3):
#     batch_data_X = 0*np.ones((batch_size,max_length,features))
#     batch_data_y = 0*np.ones((batch_size,max_length,features))
#     mask_data = np.zeros((batch_size,max_length,features))
#     batch_indices = np.random.choice(len(input_path),batch_size)
#     for i in range(batch_size):
#         signal, sample_freq = sf.read(input_path[batch_indices[i]])
#         target, sample_freq = sf.read(get_clean_file(input_path[batch_indices[i]],target_path))
#         n_window_size = int(sample_freq * window_size)
#         n_window_stride = int(sample_freq * window_stride)
#         X,_ = get_mel(signal,sample_freq, n_window_size,n_window_stride, features)
#         y,_ = get_mel(target,sample_freq, n_window_size,n_window_stride, features)
#         if X.shape[0]>max_length:
#             start = np.random.randint(X.shape[0]-max_length)
#             X=X[start:start+max_length]
#             y=y[start:start+max_length]
#             batch_data_X[i]=X
#             batch_data_y[i]=y
#         else:
#             batch_data_X[i][:X.shape[0]]=X
#             batch_data_y[i][:y.shape[0]]=y
#         mask_data[i][:y.shape[0]]=np.ones((y.shape[0],features))
#         duration= len(signal)/sample_freq
#     return batch_data_X,batch_data_y, mask_data, duration, sample_freq
def test(model,target_path,store_path):
    for p in [-10]:
        test_path = os.path.join(params["data_specs"]["noisy_path"], 'noisy_{}db'.format(p))
        # test_path = params["data_specs"]["noisy_path"]
        test_files = getListOfFiles(test_path)
        store_path = os.path.join(store_path, 'noisy_{}db'.format(p))
        print(store_path)
        denoised_dir = store_path + "/denoised/"
        noisy_dir = store_path + "/noisy/"
        clean_dir = store_path + "/clean/"
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
            # os.mkdir(clean_dir)
            os.mkdir(noisy_dir)
            os.mkdir(denoised_dir)
        for f in test_files:
            signal, sample_freq = sf.read(f)
            duration = int(len(signal) / sample_freq)
            name = f.split("/")[-1]
            target, sample_freq = sf.read(get_clean_file(f, target_path))
            y, _ = get_psf_mel(target,sample_freq, 512, 320,160, 64, pad_to=0)
            X, ph = get_psf_mel(signal,sample_freq, 512, 320,160, 64, pad_to=0)
            # x = np.expand_dims(X, axis=0)
            # x = np.expand_dims(x, axis=0)
            # x = torch.FloatTensor(x).to(device)
            # with torch.no_grad():
            #     dec = model(x)
            #     dec = dec.cpu().numpy()
            #     dec = dec.squeeze()
            # np.savez(denoised_dir + name + '.npz', features=dec, duration=duration)
            np.savez(noisy_dir + name + '.npz', features=X, duration=duration)
            np.savez(clean_dir + name + '.npz', features=y, duration=duration)

test(1,clean_path_train,"/raid/Speech/cleaned/")