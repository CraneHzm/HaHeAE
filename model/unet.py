import math
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from .blocks import *
from .graph_convolution_network import *
from .nn import (conv_nd, linear, normalization, timestep_embedding, zero_module)


@dataclass
class BeatGANsUNetConfig(BaseConfig):
    seq_len: int = 80
    in_channels: int = 9
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    out_channels: int = 9
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # number of time embed channels and style channels
    embed_channels: int = 256
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    attention_resolutions: Tuple[int] = (0, )
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4)    
    conv_resample: bool = True
    # 1 = 1d conv
    dims: int = 1    
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    use_new_attention_order: bool = False
    resnet_two_cond: bool = True
    resnet_cond_channels: int = None
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeatGANs)
    resnet_use_zero_module: bool = True

    def make_model(self):
        return BeatGANsUNetModel(self)

        
class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf

        self.dtype = th.float32

        self.time_emb_channels = conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels),
        )

        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1)),
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=conf.resnet_two_cond,
            use_zero_module=conf.resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=conf.resnet_cond_channels,
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        ds = 1
        resolution = conf.seq_len
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,                        
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)                
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,                            
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,                            
                            down=True,
                            **kwargs,
                        ).make_model() if conf.
                        resblock_updown else Downsample(ch,
                                                        conf.conv_resample,
                                                        dims=conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch
                
        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,                
                **kwargs,
            ).make_model(),
            #AttentionBlock(
            #    ch,                
            #    num_heads=conf.num_heads,
            #    num_head_channels=conf.num_head_channels,
            #    use_new_attention_order=conf.use_new_attention_order,
            #),
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,                
                **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,                        
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,                            
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,                            
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(conf.dims,
                            input_ch,
                            conf.out_channels,
                            3,  ## kernel size
                            padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1),
            )

    def forward(self, x, t, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h) ## Get output from each layer
                k += 1
        assert k == len(self.input_blocks)

        # middle blocks
        h = self.middle_block(h, emb=emb)

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


class Return(NamedTuple):
    pred: th.Tensor
    
    
@dataclass
class BeatGANsEncoderConfig(BaseConfig):    
    in_channels: int
    seq_len: int = 80
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (0, )
    model_channels: int = 32
    out_channels: int = 256
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4)
    use_time_condition: bool = False
    conv_resample: bool = True
    dims: int = 1    
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = True
    use_new_attention_order: bool = False
    pool: str = 'adaptivenonzero'

    def make_model(self):
        return BeatGANsEncoderModel(self)

        
class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    
    For usage, see UNet.
    """
    def __init__(self, conf: BeatGANsEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32

        if conf.use_time_condition: 
            time_embed_dim = conf.model_channels
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1),
                )
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = conf.seq_len
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,                        
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)               
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,                            
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            down=True,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Downsample(ch,
                                          conf.conv_resample,
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,                
            ).make_model(),
            AttentionBlock(
                ch,                
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,                
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                ## nn.AdaptiveAvgPool2d((1, 1)),
                nn.AdaptiveAvgPool1d(1),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

    def forward(self, x, t=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else: ## autoencoding.py
            emb = None

        results = []
        h = x.type(self.dtype)        
        for module in self.input_blocks: ## flow input x over all the input blocks
            h = module(h, emb=emb)                        
            if self.conf.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))                
        h = self.middle_block(h, emb=emb) ## TimestepEmbedSequential(...)
        if self.conf.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else: ## autoencoder.py
            h = h.type(x.dtype)

        h = h.float()
        h = self.out(h)

        return h

        
@dataclass
class GCNUNetConfig(BaseConfig):
    in_channels: int = 9
    node_n: int = 3
    seq_len: int = 80    
    # base channels, will be multiplied
    model_channels: int = 32
    # output of the unet
    out_channels: int = 9
    # how many repeating resblocks per resolution
    num_res_blocks: int = 8
    # number of time embed channels and style channels
    embed_channels: int = 256
    # dropout applies to the resblocks
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4)    
    resnet_two_cond: bool = True
    
    def make_model(self):
        return GCNUNetModel(self)

        
class GCNUNetModel(nn.Module):
    def __init__(self, conf: GCNUNetConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32
        assert conf.in_channels%conf.node_n == 0                
        self.in_features = conf.in_channels//conf.node_n

        self.time_emb_channels = conf.model_channels*4
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels),
        )
        
        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                graph_convolution(in_features=self.in_features, out_features=ch, node_n=conf.node_n, seq_len=conf.seq_len)),
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=conf.resnet_two_cond,
        )

        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        ds = 1
        resolution = conf.seq_len
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    residual_graph_convolution_config(
                    in_features=ch,
                    seq_len=resolution,
                    emb_channels = conf.embed_channels,
                    dropout=conf.dropout,
                    out_features=int(mult * conf.model_channels),
                    node_n=conf.node_n,
                    **kwargs,
                    ).make_model()                
                ]
                ch = int(mult * conf.model_channels)                
                self.input_blocks.append(*layers)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        graph_downsample()))
                ch = out_ch
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                layers = [
                    residual_graph_convolution_config(
                    in_features=ch + ich,
                    seq_len=resolution,
                    emb_channels = conf.embed_channels,
                    dropout=conf.dropout,
                    out_features=int(mult * conf.model_channels),
                    node_n=conf.node_n,
                    has_lateral=True if ich > 0 else False,
                    **kwargs,
                    ).make_model()
                ]
                ch = int(mult*conf.model_channels)
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(graph_upsample())
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                
        self.out = nn.Sequential(
        graph_convolution(in_features=ch, out_features=self.in_features, node_n=conf.node_n, seq_len=conf.seq_len),
        nn.Tanh(),
        )
        
    def forward(self, x, t, **kwargs):
        """
        Apply the model to an input batch.
        
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        bs, channels, seq_len = x.shape
        x = x.reshape(bs, self.conf.node_n, self.in_features, seq_len).permute(0, 2, 1, 3)
        
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))
        
        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h) ## Get output from each layer
                k += 1
        assert k == len(self.input_blocks)

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        pred = pred.permute(0, 2, 1, 3).reshape(bs, -1, seq_len)
        
        return Return(pred=pred)

        
@dataclass
class GCNEncoderConfig(BaseConfig):    
    in_channels: int
    in_features = 3 # features for one node        
    seq_len: int = 40
    seq_len_future: int = 3
    num_res_blocks: int = 2
    model_channels: int = 32
    out_channels: int = 32
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4)
    use_time_condition: bool = False
    
    def make_model(self):
        return GCNEncoderModel(self)

        
class GCNEncoderModel(nn.Module):
    def __init__(self, conf: GCNEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32
        assert conf.in_channels%conf.in_features == 0                
        self.in_features = conf.in_features
        self.node_n = conf.in_channels//conf.in_features
                        
        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            graph_convolution(in_features=self.in_features, out_features=ch, node_n=self.node_n, seq_len=conf.seq_len),
        ])
        input_block_chans = [ch]
        ds = 1
        resolution = conf.seq_len
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    residual_graph_convolution_config(
                    in_features=ch,
                    seq_len=resolution,
                    emb_channels = None,
                    dropout=conf.dropout,
                    out_features=int(mult * conf.model_channels),
                    node_n=self.node_n,
                    use_condition=conf.use_time_condition,
                    ).make_model()
                ]                
                ch = int(mult * conf.model_channels)
                self.input_blocks.append(*layers)
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    graph_downsample())
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.hand_prediction = nn.Sequential(
            conv_nd(1, ch*2, ch*2, 3, padding=1),
            nn.LayerNorm([ch*2, conf.seq_len_future]),
            nn.Tanh(),
            conv_nd(1, ch*2, self.in_features*2, 1),
            nn.Tanh(),
        )
        
        self.head_prediction = nn.Sequential(
            conv_nd(1, ch, ch, 3, padding=1),
            nn.LayerNorm([ch, conf.seq_len_future]),
            nn.Tanh(),            
            conv_nd(1, ch, self.in_features, 1),
            nn.Tanh(),
        )
        
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            conv_nd(1, ch*self.node_n, conf.out_channels, 1),
            nn.Flatten(),
        )

        
    def forward(self, x, t=None):
        bs, channels, seq_len = x.shape

        if self.node_n == 3: # both hand and head
            hand_last = x[:, :6, -1:].expand(-1, -1, self.conf.seq_len_future).clone() #last observed hand position
            head_last = x[:, 6:, -1:].expand(-1, -1, self.conf.seq_len_future).clone()# last observed head orientation

        if self.node_n == 2: # hand only
            hand_last = x[:, :, -1:].expand(-1, -1, self.conf.seq_len_future).clone() #last observed hand position
            
        if self.node_n == 1: # head only
            head_last = x[:, :, -1:].expand(-1, -1, self.conf.seq_len_future).clone()# last observed head orientation

        
        x = x.reshape(bs, self.node_n, self.in_features, seq_len).permute(0, 2, 1, 3)
        
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
                    
        h = h.type(x.dtype)
        h = h.float()
        bs, features, node_n, seq_len = h.shape
        
        if self.node_n == 3: # both hand and head
            hand_features = h[:, :, :2, -self.conf.seq_len_future:].reshape(bs, features*2, -1)        
            head_features = h[:, :, 2:, -self.conf.seq_len_future:].reshape(bs, features, -1)
            
            pred_hand = self.hand_prediction(hand_features) + hand_last
            pred_head = self.head_prediction(head_features) + head_last
            pred_head = F.normalize(pred_head, dim=1)# normalize head orientation to unit vectors
            
        if self.node_n == 2: # hand only
            hand_features = h[:, :, :, -self.conf.seq_len_future:].reshape(bs, features*2, -1)            
            pred_hand = self.hand_prediction(hand_features) + hand_last
            pred_head = None
        
        if self.node_n == 1: # head only        
            head_features = h[:, :, :, -self.conf.seq_len_future:].reshape(bs, features, -1)                        
            pred_head = self.head_prediction(head_features) + head_last
            pred_head = F.normalize(pred_head, dim=1)# normalize head orientation to unit vectors
            pred_hand = None
        
        h = h.reshape(bs, features*node_n, seq_len)
        h = self.out(h)
        
        return h, pred_hand, pred_head

        
@dataclass
class CNNEncoderConfig(BaseConfig):    
    in_channels: int    
    seq_len: int = 40
    seq_len_future: int = 3    
    out_channels: int = 128
        
    def make_model(self):
        return CNNEncoderModel(self)

        
class CNNEncoderModel(nn.Module):
    def __init__(self, conf: CNNEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32
        input_dim = conf.in_channels
        length = conf.seq_len
        out_channels = conf.out_channels
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, length]),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, length]),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, length]),
            nn.ReLU(inplace=True)            
        )
        
        self.out = nn.Linear(32 * length, out_channels)
        
    def forward(self, x, t=None):
        bs, channels, seq_len = x.shape
        hand_last = x[:, :6, -1:].expand(-1, -1, self.conf.seq_len_future).clone() #last observed hand position
        head_last = x[:, 6:, -1:].expand(-1, -1, self.conf.seq_len_future).clone()# last observed head orientation
        
        h = x.type(self.dtype)
        h = self.encoder(h)
        h = h.view(h.shape[0], -1)
        
        h = h.type(x.dtype)
        h = h.float()
        
        h = self.out(h)
        return h, hand_last, head_last

        
@dataclass
class GRUEncoderConfig(BaseConfig):    
    in_channels: int    
    seq_len: int = 40
    seq_len_future: int = 3    
    out_channels: int = 128
        
    def make_model(self):
        return GRUEncoderModel(self)

        
class GRUEncoderModel(nn.Module):
    def __init__(self, conf: GRUEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32
        input_dim = conf.in_channels
        length = conf.seq_len
        feature_channels = 32
        out_channels = conf.out_channels
        
        self.encoder = nn.GRU(input_dim, feature_channels, 1, batch_first=True)
        
        self.out = nn.Linear(feature_channels * length, out_channels)
        
    def forward(self, x, t=None):
        bs, channels, seq_len = x.shape
        hand_last = x[:, :6, -1:].expand(-1, -1, self.conf.seq_len_future).clone() #last observed hand position
        head_last = x[:, 6:, -1:].expand(-1, -1, self.conf.seq_len_future).clone()# last observed head orientation
        
        h = x.type(self.dtype)
        h, _ = self.encoder(h.permute(0, 2, 1))
        h = h.reshape(h.shape[0], -1)
        
        h = h.type(x.dtype)
        h = h.float()
        
        h = self.out(h)
        return h, hand_last, head_last

        
@dataclass
class LSTMEncoderConfig(BaseConfig):    
    in_channels: int    
    seq_len: int = 40
    seq_len_future: int = 3    
    out_channels: int = 128
        
    def make_model(self):
        return LSTMEncoderModel(self)

        
class LSTMEncoderModel(nn.Module):
    def __init__(self, conf: LSTMEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32
        input_dim = conf.in_channels
        length = conf.seq_len
        feature_channels = 32
        out_channels = conf.out_channels        
        
        self.encoder = nn.LSTM(input_dim, feature_channels, 1, batch_first=True)
                
        self.out = nn.Linear(feature_channels * length, out_channels)
        
    def forward(self, x, t=None):
        bs, channels, seq_len = x.shape
        hand_last = x[:, :6, -1:].expand(-1, -1, self.conf.seq_len_future).clone() #last observed hand position
        head_last = x[:, 6:, -1:].expand(-1, -1, self.conf.seq_len_future).clone()# last observed head orientation
        
        h = x.type(self.dtype)
        h, _ = self.encoder(h.permute(0, 2, 1))
        h = h.reshape(h.shape[0], -1)
                       
        h = h.type(x.dtype)
        h = h.float()
        
        h = self.out(h)
        return h, hand_last, head_last                

        
@dataclass
class MLPEncoderConfig(BaseConfig):    
    in_channels: int    
    seq_len: int = 40
    seq_len_future: int = 3    
    out_channels: int = 128
        
    def make_model(self):
        return MLPEncoderModel(self)

                
class MLPEncoderModel(nn.Module):
    def __init__(self, conf: MLPEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32
        input_dim = conf.in_channels
        length = conf.seq_len
        out_channels = conf.out_channels
        
        linear_size = 128
        self.encoder = nn.Sequential(
            nn.Linear(length*input_dim, linear_size),
            nn.LayerNorm([linear_size]),
            nn.ReLU(inplace=True),
            nn.Linear(linear_size, linear_size),
            nn.LayerNorm([linear_size]),
            nn.ReLU(inplace=True),
        )
                        
        self.out = nn.Linear(linear_size, out_channels)
        
    def forward(self, x, t=None):
        bs, channels, seq_len = x.shape
        hand_last = x[:, :6, -1:].expand(-1, -1, self.conf.seq_len_future).clone() #last observed hand position
        head_last = x[:, 6:, -1:].expand(-1, -1, self.conf.seq_len_future).clone()# last observed head orientation
        
        h = x.type(self.dtype)
        
        h = h.view(h.shape[0], -1)
        h = self.encoder(h)
                     
        h = h.type(x.dtype)
        h = h.float()
        
        h = self.out(h)
        return h, hand_last, head_last