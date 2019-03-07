import soundfile as sf
import numpy as np
from utils import get_mel, get_noisy
import librosa
import pickle
import torch.utils.data as data
class Dataset(data.Dataset):

    def __init__(self, filemap, features, max_length=3000,window_size=20e-3,window_stride=5e-3):
        self.filemap = filemap
        self.features = features
        self.max_length = max_length
        self.window_size = window_size
        self.window_stride = window_stride


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filemap)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mask_data = np.zeros((self.max_length, self.features))
        noisy_file = self.filemap[index][0]
        clean_file = self.filemap[index][1]
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
class DynamicDataset(data.Dataset):

    def __init__(self, original, all_noise, features, max_length=3000,window_size=20e-3,window_stride=5e-3, noise_min=-20, noise_max=-50):
        super(DynamicDataset,self).__init__()
        self.targets = original
        self.all_noise = all_noise
        self.features = features
        self.max_length = max_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.noise_min = noise_min
        self.noise_max = noise_max


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mask_data = np.zeros((self.max_length, self.features))
        noise = self.all_noise[index%len(self.all_noise)]
        target, sample_freq = sf.read(self.targets[index][0])
        snr = np.random.randint(self.noise_max, self.noise_min, 1)[0]
        percent = np.random.randint(50, 100, 1)[0]
        signal, target = get_noisy(target, noise, snr, 100)
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
