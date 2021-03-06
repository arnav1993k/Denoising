import torch
import torch.nn as nn

from apex import amp

from patter.models.model import SpeechModel
from .activation import Swish


activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "swish": Swish
}


def inf_loss_to_zero_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * (kernel_size)) // 2 - 1
    return kernel_size // 2


class JasperBlock(nn.Module):
    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1, dilation=1, padding='same', dropout=0.2, activation=None, residual=True, residual_panes=[]):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])

        layers = []
        inplanes_loop = inplanes
        for _ in range(repeat - 1):
            layers.extend(self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val))
            layers.extend(self._get_act_dropout_layer(dropout=dropout, activation=activation))
            inplanes_loop = planes
        layers.extend(self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val))
        
        self.conv = nn.Sequential(*layers)
        self.res = None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            self.res = nn.ModuleList()
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.Sequential(*self._get_conv_bn_layer(ip, planes, kernel_size=1)))
        self.out = nn.Sequential(*self._get_act_dropout_layer(dropout=dropout, activation=activation))
        self.register_backward_hook(inf_loss_to_zero_hook) # NB: only needed until the pytorch ctc loss supports this
    
    def _get_conv_bn_layer(self, inplanes, planes, kernel_size=11, stride=1, dilation=1, padding=0):
        layers = [
            nn.Conv1d(inplanes, planes, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm1d(planes, eps=1e-3)
        ]
        return layers

    def init_weights(self):
        for x in self.conv:
            if isinstance(x, nn.Conv1d):
                nn.init.xavier_normal_(x.weight)
            elif isinstance(x, nn.BatchNorm1d):
                x.reset_parameters()
        if self.res:
            for x in self.res:
                if isinstance(x, nn.Conv1d):
                    nn.init.xavier_normal_(x.weight)
                elif isinstance(x, nn.BatchNorm1d):
                    x.reset_parameters()
    
    def _get_act_dropout_layer(self, dropout=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=dropout)
        ]
        return layers

    def forward(self, xs):
        out = self.conv(xs[-1])
        if self.res is not None:
            for i in range(len(self.res)):
                out += self.res[i](xs[i])
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            return xs + [out]
        return [out]


class Jasper(SpeechModel):
    def __init__(self, cfg):
        super(Jasper, self).__init__(cfg)
        self.loss_func = None

        self.labels = cfg['labels']['labels'] + [SpeechModel.BLANK_CHAR]
        self.blank_index = len(self.labels) - 1

        activation = activations[cfg['encoder']['activation']](*cfg['encoder']['activation_params'])
        feat_in = cfg['input']['features']

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in cfg['jasper']:
            dense_res = []
            if lcfg['residual_dense']:
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            encoder_layers.append(JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'], 
                                              kernel_size=lcfg['kernel'], stride=lcfg['stride'], 
                                              dilation=lcfg['dilation'], dropout=lcfg['dropout'], 
                                              residual=lcfg['residual'], activation=activation,
                                              residual_panes=dense_res))
            feat_in = lcfg['filters']
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(
            nn.Conv1d(feat_in, len(self.labels), kernel_size=1),
            nn.LogSoftmax(dim=1))
        self.init_weights()

    def train(self, mode=True):
        """
        Enter (or exit) training mode. Initializes loss function if necessary
        :param mode: if True, set model up for training
        :return:
        """
        if mode and self.loss_func is None:
            self.loss_func = nn.CTCLoss(blank=self.blank_index, reduction='mean')
        super().train(mode=mode)

    # @amp.float_function
    def loss(self, x, y, x_length=None, y_length=None):
        """
        Compute CTC loss for the given inputs
        :param x: predicted values
        :param y: reference values
        :param x_length: length of prediction
        :param y_length: length of references
        :return:
        """
        if self.loss_func is None:
            self.train()
        return self.loss_func(x.transpose(0,1), y, x_length.to(torch.long), y_length.to(torch.long))

    def init_weights(self):
        for x in self.encoder:
            x.init_weights()
        nn.init.xavier_normal_(self.decoder[0].weight)

    def flatten_parameters(self):
        pass

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.

        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for b in self.encoder:
            try:
                for m in b.conv:
                    if type(m) == nn.modules.conv.Conv1d:
                        seq_len = ((seq_len + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) / m.stride[0] + 1)
            except Exception:
                pass
        return seq_len.int()

    def forward(self, x, lengths=0):
        """
        Perform a forward pass through the DeepSpeech model. Inputs are a batched spectrogram Variable and a Variable
        that indicates the sequence lengths of each example.

        The output (in inference mode) is a Variable containing posteriors over each character class at each timestep
        for each example in the minibatch.

        :param x: (batch_size, stft_size, max_seq_len) Raw single-channel spectrogram input
        :param lengths: (batch,) Sequence_length for each sample in batch
        :return: FloatTensor(batch_size, max_seq_len, num_classes), IntTensor(batch_size)
        """

        x = self.encoder([x])[-1]
        x = self.decoder(x)
#        output_lengths = self.get_seq_lens(lengths)

#        return x.transpose(1,2), output_lengths
        return x.transpose(1,2)

    def get_filter_images(self):
        """
        Generate a grid of images representing the convolution layer weights
        :return: list of images
        """
        images = []
        return images
