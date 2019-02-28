
import horovod.torch as hvd
import torch.utils.data as data

import torch.nn as nn
import torch.utils.data as data
from data import Dataset
from utils import *

from autoencoder import AutoEncoder_Lib

from tensorboardX import SummaryWriter

from patter.model import ModelFactory
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())
if hvd.rank() == 0:
    writer = SummaryWriter('train_summary/librosa_hvd/')
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
            "save_path":"/raid/Speech/test_models/model_lib_hvd.ckpt",
            "train_test_split":1,
            "batch_size":64,
            "num_epochs":4000,
            "device":"cuda",
            "seq_length":400 #keep odd
        }
       }
seed_model_path = "checkpoints/librosa_model.pt"
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

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.autoencoder = AutoEncoder_Lib([1,32,128,32,1]).cuda()
        self.first_layer = first_layer.cuda()
        self.loss_func = nn.KLDivLoss(reduction="batchmean")
        self.abs_loss = nn.L1Loss()
        dummy_input = torch.ones((batch_size,1,features,max_length)).cuda()
        if hvd.rank() == 0:
            writer.add_graph(self.autoencoder,dummy_input)
        self.optimizer = self.autoencoder.optimizer

    def forward(self, X, y, mask):
        X=X.cuda()
        y=y.cuda()
        mask=mask.cuda()
        op = self.autoencoder(X)
        masked_op = op * mask
        masked_y = y * mask
        masked_op = masked_op.transpose(3, 2).squeeze()
        masked_y = masked_y.transpose(3, 2).squeeze()
        dec_op = self.first_layer(masked_op)
        with torch.no_grad():
            dec_y = self.first_layer(masked_y)
        dec_op = dec_op.reshape((dec_op.shape[0] * dec_op.shape[1], dec_op.shape[-1]))
        dec_y = dec_y.reshape((dec_y.shape[0] * dec_y.shape[1], dec_op.shape[-1]))
        dec_y = torch.exp(dec_y)
        return dec_op, dec_y, masked_op, masked_y

model = Model()
# Initialize Horovod


# Define dataset...
train_dataset =  Dataset(actual_path, clean_path, features, max_length)
# training_generator = data.DataLoader(training_set, batch_size=batch_size)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(model.optimizer, named_parameters=model.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

for epoch in range(1000):
    n_iter = len(train_loader)
    lr=0.01
    sum_loss = 0
    for batch_idx, (X ,y, mask) in enumerate(train_loader):
        optimizer.zero_grad()
        prob_o, prob_y, spec_o, spec_y = model(X,y,mask)
        jasper_loss = model.loss_func(prob_o, prob_y)
        ae_loss = model.abs_loss(spec_o, spec_y)
        loss = jasper_loss+0.1*ae_loss
        sum_loss+=loss.data
        loss.backward()
        optimizer.step()
        if hvd.rank() == 0:
            writer.add_scalar('Train/Loss', loss.data, n_iter * epoch + batch_idx)
            writer.add_scalar('Train/Jasper_Loss', jasper_loss.data, n_iter * epoch + batch_idx)
            writer.add_scalar('Train/AE_Loss', ae_loss, n_iter * epoch + batch_idx)
    if epoch % 10 == 0 and epoch != 0:
        lr *= 0.9
        for param_group in model.autoencoder.optimizer.param_groups:
           param_group['lr'] = lr
    avg_loss = sum_loss / n_iter
    if hvd.rank() == 0:
        writer.add_scalar('Train/Epoch_Loss', avg_loss.data, epoch)
        print('Epoch: ', epoch, '| train loss: %.4f' % (avg_loss.data))
        if avg_loss.data < model.autoencoder.minloss:
            torch.save(model.autoencoder.state_dict(), save_path)
            model.autoencoder.minloss = avg_loss.data
        specs, ph, fname = run_test(model.autoencoder, actual_path, clean_path, features, max_length, device, psf=False)
        buf = plot_spectrogram(specs, band=300, labels=["noisy", "denoised", "target"], save=True, fname=fname)
        im_to_tensorboard(buf, writer, epoch)
        get_sound(specs, writer, ph, epoch, False, psf=False)