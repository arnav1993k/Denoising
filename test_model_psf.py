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
import time
import torch.nn as nn
import torch.utils.data as data
from data import Dataset
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

writer = SummaryWriter('train_summary/psf/')
params={
       "data_specs":{
           "noisy_path":"/raid/Speech/LibriSpeech/clean_noisy_noise/test_batch_1/input/",
           "clean_path":"/raid/Speech/LibriSpeech/clean_noise/clean/",
            },
        "spectrogram_specs":{
            "window_size":20e-3,
            "window_stride":10e-3,
            "type":"logfbank",
            "features":64
        },
        "training":{
            "save_path":"/raid/Speech/test_models/model_psf.ckpt",
            "train_test_split":1,
            "batch_size":64,
            "num_epochs":4000,
            "device":"cuda:1",
            "seq_length":304 #keep odd
        }
       }
seed_model_path = "checkpoints/psf_model.pt"
seed_model = ModelFactory.load(seed_model_path)
first_layer = seed_model
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

autoencoder = AutoEncoder([1,32,64,32,1]).to(device)
first_layer = first_layer.to(device)
loss_func = nn.KLDivLoss(reduction="batchmean")
abs_loss = nn.MSELoss()
dummy_input = torch.ones((batch_size,1,features,max_length)).to(device)
writer.add_graph(autoencoder,dummy_input)
#autoencoder.load_state_dict(torch.load(save_path))

def get_batch(batch_size, input_path, target_path, features, max_length=3000,features_type='logfbank',window_size=20e-3,window_stride=5e-3):
    batch_data_X = 0*np.ones((batch_size,max_length,features))
    batch_data_y = 0*np.ones((batch_size,max_length,features))
    mask_data = np.zeros((batch_size,max_length,features))
    batch_indices = np.random.choice(len(input_path),batch_size)
    for i in range(batch_size):
        signal, sample_freq = sf.read(input_path[batch_indices[i]])
        target, sample_freq = sf.read(get_clean_file(input_path[batch_indices[i]],target_path))
        n_window_size = int(sample_freq * window_size)
        n_window_stride = int(sample_freq * window_stride)
        X,_ = get_psf_mel(signal,sample_freq, 512, n_window_size,n_window_stride, features, pad_to=0)
        y,_ = get_psf_mel(target,sample_freq, 512, n_window_size,n_window_stride, features, pad_to=0)
        if X.shape[0]>max_length:
            start = np.random.randint(X.shape[0]-max_length)
            X=X[start:start+max_length]
            y=y[start:start+max_length]
            batch_data_X[i]=X
            batch_data_y[i]=y
        else:
            batch_data_X[i][:X.shape[0]]=X
            batch_data_y[i][:y.shape[0]]=y
        mask_data[i][:y.shape[0]]=np.ones((y.shape[0],features))
        duration= len(signal)/sample_freq
    return batch_data_X,batch_data_y, mask_data, duration, sample_freq
def train(batch_size=32, num_epochs = 1000, input_path=None, target_path=None, features=None, max_length=3003, features_type='logfbank',window_size=20e-3,window_stride=5e-3, save_path= None, verbose=False):
    lr = 0.01
    for j in range(num_epochs):
        n_iter = int(len(input_path)/batch_size)
        sum_loss = 0
        # t = 0
        for i in range(n_iter):
            # print(time.time() - t)
            # t = time.time()
        #get batch
            X,y,mask, ts,_ = get_batch(batch_size, input_path, target_path, features, max_length, features_type, window_size, window_stride)

        #convert to tensor for training
            X = np.expand_dims(X,axis=1)
            y = np.expand_dims(y,axis=1)
            mask = np.expand_dims(mask,axis=1)
            X,y = torch.FloatTensor(X).to(device),torch.tensor(y,requires_grad=False, dtype=torch.float32).to(device)
        #forward pass
            op = autoencoder(X)
            mask = torch.FloatTensor(mask).to(device)
        #mask output for segment where audio is absent
            masked_op = op*mask
            masked_y = y*mask
            masked_op = masked_op.transpose(3,2).squeeze()
            masked_y = masked_y.transpose(3,2).squeeze()
            dec_op = first_layer(masked_op)
            dec_y = first_layer(masked_y)
            dec_op = dec_op.reshape((dec_op.shape[0]*dec_op.shape[1],dec_op.shape[-1]))
            dec_y = dec_y.reshape((dec_y.shape[0]*dec_y.shape[1],dec_op.shape[-1]))
            dec_y = dec_y.detach()
            dec_y = torch.exp(dec_y)

        #Calculate loss and backprop
            jasper_loss = loss_func(dec_op, dec_y)
            ae_loss = abs_loss(masked_op,masked_y)
            loss = jasper_loss+0.1*ae_loss
            autoencoder.optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            autoencoder.optimizer.step()                    # apply gradients
            sum_loss+=loss.data
        #Log loss and save model
            # print('Iteration: ', i, '| train loss: %.4f jasper loss %.4f ae loss %' % loss.data)
            writer.add_scalar('Train/Loss', loss.data, n_iter*j+i)
            writer.add_scalar('Train/Jasper_Loss', jasper_loss.data, n_iter*j+i)
            writer.add_scalar('Train/AE_Loss',ae_loss,n_iter*j+i)
        if j%10==0 and j!=0:
            lr*=0.9
            for param_group in autoencoder.optimizer.param_groups:
                param_group['lr'] = lr
        avg_loss = sum_loss/n_iter
        writer.add_scalar('Train/Epoch_Loss',avg_loss.data,j)
        print('Epoch: ', j, '| train loss: %.4f' % (avg_loss.data))
        if loss.data<autoencoder.minloss:
            torch.save(autoencoder.state_dict(), save_path)
            autoencoder.minloss = loss.data
        specs,ph,fname = run_test(autoencoder, input_path, target_path, features, max_length, device, psf=True)
        buf = plot_spectrogram(specs,band=300,labels=["noisy","denoised","target"],save=True, fname=fname)
        im_to_tensorboard(buf,writer,j)
        get_sound(specs,writer,ph,j,False, psf=True)
# def train(loader, num_epochs=1000, input_path=None, target_path=None, save_path=None):
#     for j in range(num_epochs):
#         n_iter = len(loader)
#         i = 0
#         sum_loss = 0
#         t=0
#         for X,y,mask in loader:
#             print(time.time()-t)
#             t = time.time()
#             X, y = X.to(device), y.detach().to(device)
#             # forward pass
#             op = autoencoder(X)
#             mask = mask.to(device)
#             # mask output for segment where audio is absent
#             masked_op = op * mask
#             masked_y = y * mask
#             masked_op = masked_op.transpose(3, 2).squeeze()
#             masked_y = masked_y.transpose(3, 2).squeeze()
#             dec_op = first_layer(masked_op)
#             dec_y = first_layer(masked_y)
#             dec_op = dec_op.reshape((dec_op.shape[0] * dec_op.shape[1], dec_op.shape[-1]))
#             dec_y = dec_y.reshape((dec_y.shape[0] * dec_y.shape[1], dec_op.shape[-1]))
#             dec_y = dec_y.detach()
#             dec_y = torch.exp(dec_y)
#             # Calculate loss and backprop
#             jasper_loss = loss_func(dec_op, dec_y)
#             ae_loss = abs_loss(masked_op, masked_y)
#             loss = jasper_loss + 0.1*ae_loss
#             autoencoder.optimizer.zero_grad()  # clear gradients for this training step
#             loss.backward()  # backpropagation, compute gradients
#             autoencoder.optimizer.step()  # apply gradients
#             sum_loss += loss.data
#             # Log loss and save model
#             # print('Iteration: ', i, '| train loss: %.4f jasper loss %.4f ae loss %' % loss.data)
#             writer.add_scalar('Train/Loss', loss.data, n_iter * j + i)
#             writer.add_scalar('Train/Jasper_Loss', jasper_loss.data, n_iter * j + i)
#             writer.add_scalar('Train/AE_Loss',ae_loss,n_iter*j+i)
#             i+=1
#         avg_loss = sum_loss / n_iter
#         writer.add_scalar('Train/Epoch_Loss', avg_loss.data, j)
#         print('Epoch: ', j, '| train loss: %.4f' % (avg_loss.data))
#         if loss.data < autoencoder.minloss:
#             torch.save(autoencoder.state_dict(), save_path)
#             autoencoder.minloss = loss.data
#         specs, ph, fname = run_test(autoencoder, input_path, target_path, features, max_length, device)
#         buf = plot_spectrogram(specs, band=300, labels=["noisy", "denoised", "target"], save=True, fname=fname)
#         im_to_tensorboard(buf, writer, j)
#         get_sound(specs, writer, ph, j,gla=True)
# training_set = Dataset(actual_path, clean_path, features, max_length)
# training_generator = data.DataLoader(training_set, batch_size=batch_size)
train(batch_size, num_epochs, actual_path_train, clean_path_train, features, max_length, features_type, window_size, window_stride, save_path)
# train(training_generator, num_epochs, actual_path_train, clean_path_train, save_path)