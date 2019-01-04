import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
import math


class TfConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, groups=1, bias=True):
        super(TfConv1d, self).__init__()
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        dilation = _single(dilation)

        if padding not in ["same", "valid"]:
            raise ValueError("padding must be same or valid")
        
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        padding_lower = padding.lower()
        assert(padding_lower == "same" or padding_lower == "valid")
        
        self.padding_type = padding_lower
    
    def reset_parameters(self):
        n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        if self.padding_type == "valid":
            padding_rows = 0
            rows_odd = False
        else:
            in_rows = x.size(2)
            filter_rows = self.weight.size(2)
            out_rows = (in_rows + self.stride[0] - 1) // self.stride[0]
            padding_rows = max(0, (out_rows - 1) * self.stride[0] + (filter_rows - 1) * self.dilation[0] + 1 - in_rows)
            rows_odd = (padding_rows % 2) != 0
        if rows_odd:
            x = F.pad(x, (0,1), "constant", 0)
        
        return F.conv1d(x, self.weight, self.bias, self.stride,
                        padding_rows // 2, self.dilation, self.groups)
    
