import soundfile as sf
import numpy as np
from utils import get_psf_mel, get_clean_file, getListOfFiles, get_mel
import torch
import torch.utils.data as data
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
        X,_ = get_psf_mel(signal,sample_freq, 512, n_window_size,n_window_stride, features)
        y,_ = get_psf_mel(target,sample_freq, 512, n_window_size,n_window_stride, features)
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

class Dataset(data.Dataset):

    def __init__(self, input_path, target_path, features, max_length=3000,window_size=20e-3,window_stride=5e-3):
        self.input_path = input_path
        self.target_path = target_path
        self.features = features
        self.max_length = max_length
        self.window_size = window_size
        self.window_stride = window_stride


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_path)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mask_data = np.zeros((self.max_length, self.features))
        noisy_file = self.input_path[index]
        clean_file = get_clean_file(noisy_file,self.target_path)
        signal, sample_freq = sf.read(noisy_file)
        target, sample_freq = sf.read(clean_file)
        n_window_size = int(sample_freq * self.window_size)
        n_window_stride = int(sample_freq * self.window_stride)
        X, _ = get_mel(signal, sample_freq, 512, n_window_size, n_window_stride, self.features)
        y, _ = get_mel(target, sample_freq, 512, n_window_size, n_window_stride,self.features)
        if X.shape[0] > self.max_length:
            start = np.random.randint(X.shape[0] - self.max_length)
            X = X[start:start + self.max_length]
            y = y[start:start + self.max_length]
            mask_data = np.ones((self.max_length, self.features))
        else:
            X = np.pad(X, [(0, self.max_length-X.shape[0]),(0,0)], mode='constant',constant_values=-5)
            mask_data[:y.shape[0]] = np.ones((y.shape[0], self.features))
            y = np.pad(y, [(0, self.max_length - y.shape[0]), (0, 0)], mode='constant', constant_values=-5)
        X = np.expand_dims(X, axis=0).astype(np.float32)
        y = np.expand_dims(y, axis=0).astype(np.float32)
        mask_data = np.expand_dims(mask_data, axis=0).astype(np.float32)
        return X, y, mask_data

# actual_path = sorted(getListOfFiles("/raid/Speech/LibriSpeech/clean_noisy_noise/test_batch_1/input/"))
# clean_path = sorted(getListOfFiles("/raid/Speech/LibriSpeech/clean_noise/clean/"))
# training_set = Dataset(actual_path, clean_path, 64, 30001)
# training_generator = data.DataLoader(training_set, batch_size=32)
# training_set.__getitem__(1)