from __future__ import print_function
import argparse
import torch.nn as nn
import csv
from data import DynamicDataset
from utils import *

from autoencoder import AutoEncoder_Lib_Noise

from tensorboardX import SummaryWriter

from patter.model import ModelFactory


# Pin GPU to be used to process local rank (one GPU per process)

#=====START: ADDED FOR DISTRIBUTED======
'''Add custom module for distributed'''

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

'''Import distributed data loader'''
import torch.utils.data
import torch.utils.data.distributed

'''Import torch.distributed'''
import torch.distributed as dist
import pickle
#=====END:   ADDED FOR DISTRIBUTED======

# Training settings
parser = argparse.ArgumentParser(description='Denoiser')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--ngc', type=bool, default=False,
                    help='train on NGC')
parser.add_argument('--noise_min', type=int, default=0,
                    help='noise level minimum')
parser.add_argument('--noise_max', type=bool, default=-15,
                    help='noise level maximum')
#======START: ADDED FOR DISTRIBUTED======
'''
Add some distributed options. For explanation of dist-url and dist-backend please see
http://pytorch.org/tutorials/intermediate/dist_tuto.html

--local_rank will be supplied by the Pytorch launcher wrapper (torch.distributed.launch)
'''
parser.add_argument("--local_rank", default=0, type=int)

#=====END:   ADDED FOR DISTRIBUTED======

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#======START: ADDED FOR DISTRIBUTED======
'''Add a convenience flag to see if we are running distributed'''
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    '''Check that we are running with cuda, as distributed is only supported for cuda.'''
    assert args.cuda, "Distributed mode requires running with CUDA."

    '''
    Set cuda device so everything is done on the right GPU.
    THIS MUST BE DONE AS SOON AS POSSIBLE.
    '''
    torch.cuda.set_device(args.local_rank)

    '''Initialize distributed communication'''
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

#=====END:   ADDED FOR DISTRIBUTED======

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#=====START: ADDED FOR DISTRIBUTED======
'''
Change sampler to distributed if running distributed.
Shuffle data loader only if distributed.
'''
if args.ngc:
    params={
           "data_specs":{
               "noisy_path":"/noisy/",
               "clean_path":"/data/librispeech/librivox-train-clean-100.csv",
                },
            "spectrogram_specs":{
                "window_size":20e-3,
                "window_stride":10e-3,
                "type":"logfbank",
                "features":64
            },
            "training":{
                "save_path":"/results/model_lib_full_apx.ckpt",
                "train_test_split":1,
                "batch_size":64,
                "num_epochs":4000,
                "device":"cuda",
                "seq_length":400,
                "seed_model_path":"/raid/jasper_torch/librosa_model.pt",
                "summary_path":"/results/"
            }
           }
else:
    params = {
        "data_specs": {
            "noisy_path": "/raid/Speech/Natural Noise/sounds/village",
            "clean_path": "/raid/Speech/LibriSpeech/dev-clean-wav-16KHz/noisy.csv",
            "file_map_path": "/raid/Speech/file_map.pkl"
        },
        "spectrogram_specs": {
            "window_size": 20e-3,
            "window_stride": 10e-3,
            "type": "logfbank",
            "features": 64
        },
        "training": {
            "save_path": "/raid/Speech/test_models/model_lib_apx.ckpt",
            "train_test_split": 1,
            "batch_size": 64,
            "num_epochs": 4000,
            "device": "cuda",
            "seq_length": 400,
            "seed_model_path": "checkpoints/librosa_model.pt",
            "summary_path": "train_summary/librosa_apx/"
        }
    }
if args.local_rank == 0:
    writer = SummaryWriter(params["training"]["summary_path"])
seed_model_path = params["training"]["seed_model_path"]
seed_model = ModelFactory.load(seed_model_path)
first_layer = seed_model
features = params["spectrogram_specs"]["features"]

#train config
device = torch.device(params["training"]["device"])
max_length=params["training"]["seq_length"]
batch_size = args.batch_size
num_epochs = params["training"]["num_epochs"]
save_path = params["training"]["save_path"]
split = params["training"]["train_test_split"]
noise_min = args.noise_min
noise_max = args.noise_max
#data config
noisy_files = sorted(getListOfFiles(params["data_specs"]["noisy_path"]))
all_noise = []
for n in noisy_files:
    noise,sr = librosa.load(n, sr=16000)
    all_noise.append(noise)

#Load namelist clean_files
csv_path = params["data_specs"]["clean_path"]
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    clean_files = list(reader)
clean_files = clean_files[1:]
split_point = int(split*len(clean_files))
actual_path_train = clean_files[:split_point]
clean_path_train = clean_files[:split_point]
actual_path_test = clean_files[split_point:]
clean_path_test = clean_files[split_point:]

#spectrogram config
features_type=params["spectrogram_specs"]["type"]
window_size=params["spectrogram_specs"]["window_size"]
window_stride=params["spectrogram_specs"]["window_stride"]

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.autoencoder = AutoEncoder_Lib_Noise([1,32,64,64,32,1]).cuda()
        self.first_layer = first_layer.cuda()
        self.loss_func = nn.KLDivLoss(reduction="batchmean")
        self.abs_loss = nn.L1Loss()
        if args.local_rank==0:
            dummy_input = torch.ones((batch_size, 1, features, max_length)).cuda()
            writer.add_graph(self.autoencoder,dummy_input)
        self.optimizer = self.autoencoder.optimizer

    def forward(self, X, y, mask,noise):
        X=X.cuda()
        y=y.cuda()
        mask=mask.cuda()
        op,n = self.autoencoder(X)
        masked_op = op * mask
        masked_y = y * mask
        masked_n = (X-n) * mask
        masked_op = masked_op.transpose(3, 2).squeeze()
        masked_y = masked_y.transpose(3, 2).squeeze()
        masked_n = masked_n.transpose(3, 2).squeeze()
        dec_op = self.first_layer(masked_op)
        dec_n = self.first_layer(masked_n)
        with torch.no_grad():
            dec_y = self.first_layer(masked_y)
        dec_op = dec_op.reshape((dec_op.shape[0] * dec_op.shape[1], dec_op.shape[-1]))
        dec_y = dec_y.reshape((dec_y.shape[0] * dec_y.shape[1], dec_op.shape[-1]))
        dec_n = dec_n.reshape((dec_n.shape[0] * dec_n.shape[1], dec_op.shape[-1]))
        # dec_y = torch.argmax(dec_y,dim=1)
        print(dec_op.shape,dec_n.shape, dec_y.shape)
        dec_y = torch.exp(dec_y)
        return dec_op, dec_y, masked_op, masked_y, dec_n

# Define dataset...
train_dataset =  DynamicDataset(actual_path_train, all_noise, features, max_length,noise_min=noise_min,noise_max=noise_max,return_noise=True)

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler,
    batch_size=batch_size, shuffle=(train_sampler is None), **kwargs
)

#=====END:   ADDED FOR DISTRIBUTED======

model = Model()
if args.cuda:
    model.cuda()

#=====START: ADDED FOR DISTRIBUTED======
'''
Wrap model in our version of DistributedDataParallel.
This must be done AFTER the model is converted to cuda.
'''

if args.distributed:
    model = DDP(model)
#=====END:   ADDED FOR DISTRIBUTED======

optimizer = model.module.autoencoder.optimizer

def train(epoch):
    n_iter = len(train_loader)
    lr=0.01
    sum_loss = 0
    for batch_idx, (X ,y, mask,n) in enumerate(train_loader):
        optimizer.zero_grad()
        prob_o, prob_y, spec_o, spec_y, prob_n = model(X,y,mask,n)
        jasper_loss = model.module.loss_func(prob_o, prob_y)
        noise_loss = model.module.loss_func(prob_n, prob_y)
        ae_loss = model.module.abs_loss(spec_o, spec_y)
        loss = jasper_loss+noise_loss+ae_loss
        sum_loss+=loss.data
        loss.backward()
        optimizer.step()
        if args.local_rank == 0:
            writer.add_scalar('Train/Noise Loss', noise_loss.data, n_iter * epoch + batch_idx)
            writer.add_scalar('Train/Loss', loss.data, n_iter * epoch + batch_idx)
            writer.add_scalar('Train/Jasper_Loss', jasper_loss.data, n_iter * epoch + batch_idx)
            writer.add_scalar('Train/AE_Loss', ae_loss, n_iter * epoch + batch_idx)
            print("Step {} of epoch {} finished.".format(batch_idx,epoch))
    if epoch % 10 == 0 and epoch != 0:
        lr *= 0.9
        for param_group in model.module.autoencoder.optimizer.param_groups:
           param_group['lr'] = lr
    avg_loss = sum_loss / n_iter
    if args.local_rank == 0:
        writer.add_scalar('Train/Epoch_Loss', avg_loss.data, epoch)
        print('Epoch: ', epoch, '| train loss: %.4f' % (avg_loss.data))
        if avg_loss.data < model.module.autoencoder.minloss:
            torch.save(model.module.autoencoder.state_dict(), save_path)
            model.module.autoencoder.minloss = avg_loss.data
        idx = np.random.choice(len(actual_path_train), 1)[0]
        target, sr = sf.read(actual_path_train[idx][0])
        noise = all_noise[idx % len(all_noise)]
        level = np.random.randint(noise_max, noise_min, 1)[0]
        signal, target = get_noisy(target, noise, level, 100)
        specs, ph = run_test(model.module.autoencoder, signal, target, sr, features, max_length, device, psf=False)
        buf = plot_spectrogram(specs, band=300, labels=["noisy", "denoised", "noise", "target"], save=True, fname=actual_path_train[idx][0]+" {} db noise".format(level))
        im_to_tensorboard(buf, writer, epoch)
        get_sound(specs, writer, ph, epoch, False, psf=False,text=["Original","Denoised","Noise","Target"])




for epoch in range(0, args.epochs):
    #=====START: ADDED FOR DISTRIBUTED======
    if args.distributed:
        train_sampler.set_epoch(epoch)
    #=====END:   ADDED FOR DISTRIBUTED======

    train(epoch)
