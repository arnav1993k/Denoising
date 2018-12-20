import scipy
import torch
import librosa
import python_speech_features as psf
from .perturb import AudioAugmentor
from .segment import AudioSegment

windows = {
    'hann': scipy.signal.hann,
    'hamming': scipy.signal.hamming,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett,
    'none': None,
}

class LogFilterbankFeaturizer(object):
    def __init__(self, input_cfg, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.cfg = input_cfg
        self.window = windows.get(self.cfg['window'], None)

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path):
        audio = AudioSegment.from_file(file_path, target_sr=self.cfg['sample_rate'], int_values=self.cfg['int_values'])
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)

        n_fft = self.cfg['n_fft'] if self.cfg['n_fft'] is not None else int(self.cfg['sample_rate'] * self.cfg['window_size'])

        feats = psf.logfbank(signal=audio_segment.samples, samplerate=self.cfg['sample_rate'], winlen=self.cfg['window_size'],
                             winstep=self.cfg['window_stride'], nfilt=self.cfg['features'], nfft=n_fft, lowfreq=0,
                             highfreq=self.cfg['sample_rate']/2, preemph=0.97)
        spect = torch.tensor(feats,dtype=torch.float)
        if self.cfg['normalize']:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        return cls(input_config, augmentor=aa)


class LogSpectrogramFeaturizer(object):
    def __init__(self, input_cfg, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.cfg = input_cfg
        self.window = windows.get(self.cfg['window'], windows['hamming'])

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path):
        audio = AudioSegment.from_file(file_path, target_sr=self.cfg['sample_rate'], int_values=self.cfg['int_values'])
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)

        n_fft = int(self.cfg['sample_rate'] * self.cfg['window_size'])
        hop_length = int(self.cfg['sample_rate'] * self.cfg['window_stride'])
        dfft = librosa.stft(audio_segment.samples, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=self.window)
        spect, _ = librosa.magphase(dfft)
        spect = torch.FloatTensor(spect).log1p()
        if self.cfg['normalize']:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        return cls(input_config, augmentor=aa)

class FeaturizerFactory(object):
    featurizers = {
        "logfbank": LogFilterbankFeaturizer,
        "logspect": LogSpectrogramFeaturizer
    }

    def __init__(self):
        pass

    @classmethod
    def from_config(cls, input_cfg, perturbation_configs=None):
        feat_type = input_cfg['feat_type']
        featurizer = cls.featurizers[feat_type]
        return featurizer.from_config(input_cfg, perturbation_configs=perturbation_configs)
    