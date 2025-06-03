import torch.nn as nn
import torch
from dataclasses import dataclass
from torch.nn.parameter import Parameter
from numbers import Number
import torch.nn.functional as F
from .blocks import *
import math


class graph_convolution(nn.Module):
    def __init__(self, in_features, out_features, node_n = 3, seq_len = 80, bias=True):
        super(graph_convolution, self).__init__()
        
        self.temporal_graph_weights = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.feature_weights = Parameter(torch.FloatTensor(in_features, out_features))
        self.spatial_graph_weights = Parameter(torch.FloatTensor(node_n, node_n))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.spatial_graph_weights.size(1))
        self.feature_weights.data.uniform_(-stdv, stdv)
        self.temporal_graph_weights.data.uniform_(-stdv, stdv)
        self.spatial_graph_weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        y = torch.matmul(x, self.temporal_graph_weights)
        y = torch.matmul(y.permute(0, 3, 2, 1), self.feature_weights)    
        y = torch.matmul(self.spatial_graph_weights, y).permute(0, 3, 2, 1).contiguous()
        
        if self.bias is not None:
            return (y + self.bias)
        else:
            return y

            
@dataclass
class residual_graph_convolution_config():
    in_features: int
    seq_len: int
    emb_channels: int
    dropout: float
    out_features: int = None
    node_n: int = 3
    # condition the block with time (and encoder's output)
    use_condition: bool = True    
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None    
    has_lateral: bool = False
    graph_convolution_bias: bool = True
    scale_bias: float = 1
    
    def __post_init__(self):
        self.out_features = self.out_features or self.in_features
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return residual_graph_convolution(self)
        
        
class residual_graph_convolution(TimestepBlock):
    def __init__(self, conf: residual_graph_convolution_config):
        super(residual_graph_convolution, self).__init__()
        self.conf = conf        
        
        self.gcn = graph_convolution(conf.in_features, conf.out_features, node_n=conf.node_n, seq_len=conf.seq_len, bias=conf.graph_convolution_bias)        
        self.ln = nn.LayerNorm([conf.out_features, conf.node_n, conf.seq_len])
        self.act_f = nn.Tanh()
        self.dropout = nn.Dropout(conf.dropout)
        
        if conf.use_condition:
            # condition layers for the out_layers
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conf.emb_channels, conf.out_features),
            )

            if conf.two_cond:
                self.cond_emb_layers = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(conf.cond_emb_channels, conf.out_features),
                )
        
        if conf.in_features == conf.out_features:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Sequential(
                    graph_convolution(conf.in_features, conf.out_features, node_n=conf.node_n, seq_len=conf.seq_len, bias=conf.graph_convolution_bias),
                    nn.Tanh(),
                )
                                          
    def forward(self, x, emb=None, cond=None, lateral=None):
        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if "has_lateral"
            assert lateral is not None
            x = torch.cat((x, lateral), dim =1)

        y = self.gcn(x)
        y = self.ln(y)
        
        if self.conf.use_condition:
            if emb is not None:
                emb = self.emb_layers(emb).type(x.dtype)
                # adjusting shapes
                while len(emb.shape) < len(y.shape):            
                    emb = emb[..., None]
            
            if self.conf.two_cond or True:
                if cond is not None:
                    if not isinstance(cond, torch.Tensor):
                        assert isinstance(cond, dict)
                        cond = cond['cond']
                    cond = self.cond_emb_layers(cond).type(x.dtype)
                    while len(cond.shape) < len(y.shape):
                        cond = cond[..., None]
                scales = [emb, cond]
            else:
                scales = [emb]
                
            # condition scale bias could be a list
            if isinstance(self.conf.scale_bias, Number):
                biases = [self.conf.scale_bias] * len(scales)
            else:
                # a list
                biases = self.conf.scale_bias
                
            # scale for each condition
            for i, scale in enumerate(scales):
                # if scale is None, it indicates that the condition is not provided
                if scale is not None:                        
                    y = y*(biases[i] + scale)
                                
        y = self.act_f(y)
        y = self.dropout(y)        
        return self.skip_connection(x) + y

        
class graph_downsample(nn.Module):
    """
    A downsampling layer   
    """
    def __init__(self, kernel_size = 2):
        super().__init__()
        self.downsample = nn.AvgPool1d(kernel_size = kernel_size)
        
    def forward(self, x):               
        bs, features, node_n, seq_len = x.shape
        x = x.reshape(bs, features*node_n, seq_len)
        x = self.downsample(x)
        x = x.reshape(bs, features, node_n, -1)
        return x

        
class graph_upsample(nn.Module):
    """
    An upsampling layer
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):           
        x = F.interpolate(x, (x.shape[2], x.shape[3]*self.scale_factor), mode="nearest")
        return x