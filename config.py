from model.blocks import *
from diffusion.resample import UniformSampler
from dataclasses import dataclass
from diffusion.diffusion import space_timesteps
from typing import Tuple
from config_base import BaseConfig
from diffusion import *
from diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from model import *
from choices import *
from preprocess import *
import os

@dataclass
class TrainConfig(BaseConfig):
    name: str = ''
    base_dir: str = './checkpoints/'
    logdir: str = f'{base_dir}{name}'
    data_name: str = ''
    data_val_name: str = ''
    seq_len: int = 40 # for reconstruction
    seq_len_future: int = 3 # for prediction
    in_channels = 9
    fp16: bool = True
    lr: float = 1e-4
    ema_decay: float = 0.9999
    seed: int = 0 # random seed
    batch_size: int = 64
    accum_batches: int = 1
    batch_size_eval: int = 1024
    total_epochs: int = 1_000
    save_every_epochs: int = 10
    eval_every_epochs: int = 10
    train_mode: TrainMode = TrainMode.diffusion
    T: int = 1000
    T_eval: int = 100
    diffusion_type: str = 'beatgans'
    semantic_encoder_type: str = 'gcn'
    net_beatgans_embed_channels: int = 128
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    hand_mse_factor = 1.0
    head_mse_factor = 1.0
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    beta_scheduler: str = 'linear'    
    net_ch: int = 64
    net_ch_mult: Tuple[int, ...]= (1, 2, 4)    
    net_enc_channel_mult: Tuple[int]  = (1, 2, 4)
    grad_clip: float = 1
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    warmup: int = 0
    model_conf: ModelConfig = None
    model_name: ModelName = ModelName.beatgans_autoenc
    model_type: ModelType = None
    
    @property
    def batch_size_effective(self):
        return self.batch_size*self.accum_batches
        
    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'beatgans':
            # can use T < self.T for evaluation
            # follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=T, section_counts=section_counts),
                fp16=self.fp16,
            )
        else:
            raise NotImplementedError()

    @property
    def model_out_channels(self):
        return self.in_channels
    
    @property
    def model_input_channels(self):
        return self.in_channels
        
    def make_T_sampler(self):        
        return UniformSampler(self.T)

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_model_conf(self):
        cls = BeatGANsAutoencConfig
        if self.model_name == ModelName.beatgans_autoenc:
            self.model_type = ModelType.autoencoder
        else:
            raise NotImplementedError()
            
        self.model_conf = cls(
            semantic_encoder_type = self.semantic_encoder_type,
            channel_mult=self.net_ch_mult,
            seq_len = self.seq_len,
            seq_len_future = self.seq_len_future,
            embed_channels=self.net_beatgans_embed_channels,
            enc_out_channels=self.net_beatgans_embed_channels,
            enc_channel_mult=self.net_enc_channel_mult,                
            in_channels=self.model_input_channels,
            model_channels=self.net_ch,            
            out_channels=self.model_out_channels,                
        )
        
        return self.model_conf
        
def egobody_autoenc(mode, encoder_type='gcn', hand_mse_factor=1.0, head_mse_factor=1.0, data_sample_rate=1, epoch=130,in_channels=9, seq_len=40):
    conf = TrainConfig()
    conf.seq_len = seq_len
    conf.seq_len_future = 3
    conf.in_channels = in_channels
    conf.net_beatgans_embed_channels = 128
    conf.net_ch = 64
    conf.net_ch_mult = (1, 1, 1)
    conf.semantic_encoder_type = encoder_type
    conf.hand_mse_factor = hand_mse_factor
    conf.head_mse_factor = head_mse_factor   
    conf.net_enc_channel_mult = conf.net_ch_mult
    conf.total_epochs = epoch
    conf.save_every_epochs = 10
    conf.eval_every_epochs = 10
    conf.batch_size = 64
    conf.batch_size_eval = 1024*4
    
    conf.data_dir = "/scratch/hu/pose_forecast/egobody_pose2gaze/"
    conf.data_sample_rate = data_sample_rate
    conf.name = 'egobody_autoenc'
    conf.data_name = 'egobody'
    
    conf.mode = mode
    conf.make_model_conf()
    return conf