from enum import Enum

import torch, pdb
import os
from torch import Tensor
from torch.nn.functional import silu
from .unet import *
from choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    seq_len_future: int = 3
    enc_out_channels: int = 128
    semantic_encoder_type: str = 'gcn'
    enc_channel_mult: Tuple[int] = None    
    def make_model(self):
        return BeatGANsAutoencModel(self)
        
class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf
        
        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )
        
        if conf.semantic_encoder_type == 'gcn':
            self.encoder = GCNEncoderConfig(
                seq_len=conf.seq_len,
                seq_len_future=conf.seq_len_future,
                in_channels=conf.in_channels,
                model_channels=16,
                out_channels=conf.enc_out_channels,            
                channel_mult=conf.enc_channel_mult or conf.channel_mult,
            ).make_model()
        elif conf.semantic_encoder_type == '1dcnn':
            self.encoder = CNNEncoderConfig(
                seq_len=conf.seq_len,
                seq_len_future=conf.seq_len_future,
                in_channels=conf.in_channels,
                out_channels=conf.enc_out_channels,
            ).make_model()
        elif conf.semantic_encoder_type == 'gru':
            # ensure deterministic behavior of RNNs
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
            self.encoder = GRUEncoderConfig(
                seq_len=conf.seq_len,
                seq_len_future=conf.seq_len_future,
                in_channels=conf.in_channels,
                out_channels=conf.enc_out_channels,
            ).make_model()            
        elif conf.semantic_encoder_type == 'lstm':
            # ensure deterministic behavior of RNNs
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"        
            self.encoder = LSTMEncoderConfig(
                seq_len=conf.seq_len,
                seq_len_future=conf.seq_len_future,
                in_channels=conf.in_channels,
                out_channels=conf.enc_out_channels,
            ).make_model()            
        elif conf.semantic_encoder_type == 'mlp':
            self.encoder = MLPEncoderConfig(
                seq_len=conf.seq_len,
                seq_len_future=conf.seq_len_future,
                in_channels=conf.in_channels,
                out_channels=conf.enc_out_channels,
            ).make_model()
        else:
            raise NotImplementedError()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        cond, pred_hand, pred_head = self.encoder.forward(x)
        return cond, pred_hand, pred_head

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S
            
    def forward(self,
                x,
                t,                
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        if t_cond is None:
            t_cond = t ## randomly sampled timestep with the size of [batch_size]

        if noise is not None: 
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        cond_given = True
        if cond is None:
            cond_given = False
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

            cond, pred_hand, pred_head = self.encode(x_start)
            
        if t is not None: ## t==t_cond
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            #print("t: {}, _t_emb:{}".format(t, _t_emb))
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
            #print("t_cond: {}, _t_cond_emb:{}".format(t, _t_cond_emb))
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward( ## self.time_embed is an MLP
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style ## style==None, res.style: cond, torch.Size([64, 512])
        

        # where in the model to supply time conditions
        enc_time_emb = emb ## time embeddings
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb ## z_sem embeddings
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb        

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)
            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)
                    # print(i, j, h.shape)
                    '''if h.shape[-1]%2==1:
                        pdb.set_trace()'''
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]
            pdb.set_trace()
        
        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop() ## in the reverse order (symmetric)
                except IndexError:
                    lateral = None
                '''print(i, j, lateral.shape, h.shape)
                if lateral.shape[-1]!=h.shape[-1]:
                    pdb.set_trace()'''
                # print("h is", h.size())
                # print("lateral is", lateral.size())
                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1
                
        pred = self.out(h)
        # print("h:", h.shape)
        # print("pred:", pred.shape)
        
        if cond_given == True:
            return AutoencReturn(pred=pred, cond=cond)
        else:
            return AutoencReturn(pred=pred, cond=cond, pred_hand=pred_hand, pred_head=pred_head)

        
@dataclass
class GCNAutoencConfig(GCNUNetConfig):
    # number of style channels
    enc_out_channels: int = 256
    enc_channel_mult: Tuple[int] = None
    def make_model(self):
        return GCNAutoencModel(self)
        
        
class GCNAutoencModel(GCNUNetModel):
    def __init__(self, conf: GCNAutoencConfig):
        super().__init__(conf)
        self.conf = conf
        
        # having only time, cond
        self.time_emb_channels = conf.model_channels
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=self.time_emb_channels,
            time_out_channels=conf.embed_channels,
        )
        
        self.encoder = GCNEncoderConfig(
            seq_len=conf.seq_len,
            in_channels=conf.in_channels,
            model_channels=32,
            out_channels=conf.enc_out_channels,            
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
        ).make_model()

    def encode(self, x):
        cond = self.encoder.forward(x)
        return {'cond': cond}
        
    def forward(self,
                x,
                t,
                x_start=None,
                cond=None,                
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original input to encode
            cond: output of the encoder
        """
        
        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

            tmp = self.encode(x_start) 
            cond = tmp['cond'] 
            
        if t is not None: 
            _t_emb = timestep_embedding(t, self.time_emb_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward( ## self.time_embed is an MLP
                time_emb=_t_emb,
                cond=cond,
            )
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb            
        else:
            raise NotImplementedError()
       
        # where in the model to supply time conditions
        enc_time_emb = emb ## time embeddings
        mid_time_emb = emb
        dec_time_emb = emb
        enc_cond_emb = cond_emb ## z_sem embeddings
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb
        
        
        bs, channels, seq_len = x.shape
        x = x.reshape(bs, self.conf.node_n, self.in_features, seq_len).permute(0, 2, 1, 3)        
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        h = x.type(self.dtype)
        
        # input blocks
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):                
                h = self.input_blocks[k](h,
                                         emb=enc_time_emb,
                                         cond=enc_cond_emb)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)
                
        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop() ## in the reverse order (symmetric)
                except IndexError:
                    lateral = None
                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1


        pred = self.out(h)
        pred = pred.permute(0, 2, 1, 3).reshape(bs, -1, seq_len)

        return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None
    pred_hand: Tensor = None
    pred_head: Tensor = None

    
class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond) ## style==cond
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)