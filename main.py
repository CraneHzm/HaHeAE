import warnings
warnings.filterwarnings("ignore")
import os
os.nice(5)
import copy, wandb
from tqdm import tqdm, trange
import argparse
import json
import re
import random 
import math
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.optim.optimizer import Optimizer
from config import *
import time
import datetime

class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()

        ## wandb
        self.save_hyperparameters({k:v for (k,v) in vars(conf).items() if not callable(v)})
        
        if conf.seed is not None:
            pl.seed_everything(conf.seed)
            
        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf
        self.model = conf.make_model_conf().make_model()

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.3f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        self.T_sampler = conf.make_T_sampler()
        self.save_every_epochs = conf.save_every_epochs
        self.eval_every_epochs = conf.eval_every_epochs
    
    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed
            np.random.seed(seed)
            random.seed(seed)  # Python random module.
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print('seed:', seed)
        ##############################################
        
        ## Load dataset
        if self.conf.mode == 'train':
            self.train_data = load_egobody(self.conf.data_dir, self.conf.seq_len+self.conf.seq_len_future, self.conf.data_sample_rate, train=1)
            self.val_data = load_egobody(self.conf.data_dir, self.conf.seq_len+self.conf.seq_len_future, self.conf.data_sample_rate, train=0)
            if self.conf.in_channels == 6: # hand only
                self.train_data = self.train_data[:, :6, :]
                self.val_data = self.val_data[:, :6, :]
            if self.conf.in_channels == 3: # head only
                self.train_data = self.train_data[:, 6:, :]
                self.val_data = self.val_data[:, 6:, :]
                        
    def encode(self, x):
        assert self.conf.model_type.has_autoenc()
        cond, pred_hand, pred_head = self.ema_model.encoder.forward(x)
        return cond, pred_hand, pred_head

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else: 
            sampler = self.conf._make_diffusion_conf(T).make_sampler() # get noise at step T

        ## x_0 -> x-T using reverse of inference
        out = sampler.ddim_reverse_sample_loop(self.ema_model, x, model_kwargs={'cond': cond})
        ''' 'sample': x_T
            'sample_t': x_t, t in (1, ..., T)
            'xstart_t': predicted x_0 at each timestep. "xstart here is a bit different from sampling from T = T-1 to T = 0"
            'T': (1, ..., T)
        '''
        return out['sample']
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.conf.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.conf.batch_size_eval, shuffle=False)
        
    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            x_start = batch[:, :, :self.conf.seq_len]            
            x_future = batch[:, :, self.conf.seq_len:]            
            
            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                ''' self.T_sampler: diffusion.resample.UniformSampler (weights for all timesteps are 1)
                 - t: a tensor of timestep indices.
                 - weight: a tensor of weights to scale the resulting losses.
                 ## num_timesteps is self.conf.T == 1000
                '''
                losses = self.sampler.training_losses(model=self.model,
                                                      x_start=x_start,
                                                      t=t,
                                                      x_future=x_future,
                                                      hand_mse_factor = self.conf.hand_mse_factor,
                                                      head_mse_factor = self.conf.head_mse_factor,                                                      
                                                      )
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean() ## average loss across mini-batches
            #noise_mse = losses['mse'].mean()
            #hand_mse = losses['hand_mse'].mean()
            #head_mse = losses['head_mse'].mean()
            
        ## Log loss and metric (wandb)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        #self.log("train_noise_mse", noise_mse, on_epoch=True, prog_bar=True)
        #self.log("train_hand_mse", hand_mse, on_epoch=True, prog_bar=True)
        #self.log("train_head_mse", head_mse, on_epoch=True, prog_bar=True)
        return {'loss': loss} 
        
    def validation_step(self, batch, batch_idx):
        if self.conf.in_channels == 9: # both hand and head
            if((self.current_epoch+1)% self.eval_every_epochs == 0):
                batch_future = batch[:, :, self.conf.seq_len:]
                gt_hand_future = batch_future[:, :6, :]
                gt_head_future = batch_future[:, 6:, :]
                batch = batch[:, :, :self.conf.seq_len]
                cond, pred_hand_future, pred_head_future = self.encode(batch)
                xT = self.encode_stochastic(batch, cond)
                pred_xstart = self.generate(xT, cond)
                
                # hand reconstruction error
                gt_hand = batch[:, :6, :]
                pred_hand = pred_xstart[:, :6, :]                        
                bs, channels, seq_len = gt_hand.shape
                gt_hand = gt_hand.reshape(bs, 2, 3, seq_len)
                pred_hand = pred_hand.reshape(bs, 2, 3, seq_len)            
                hand_traj = torch.mean(torch.norm(gt_hand - pred_hand, dim=2))
                
                # hand prediction error
                bs, channels, seq_len = gt_hand_future.shape
                gt_hand_future = gt_hand_future.reshape(bs, 2, 3, seq_len)            
                pred_hand_future = pred_hand_future.reshape(bs, 2, 3, seq_len)
                baseline_hand_future = gt_hand[:, :, :, -1:].expand(-1, -1, -1, self.conf.seq_len_future).clone()
                hand_traj_future = torch.mean(torch.norm(gt_hand_future - pred_hand_future, dim=2))
                hand_traj_future_baseline = torch.mean(torch.norm(gt_hand_future - baseline_hand_future, dim=2))
                
                # head reconstruction error
                gt_head = batch[:, 6:, :]
                gt_head = F.normalize(gt_head, dim=1) # normalize head orientation to unit vectors
                pred_head = pred_xstart[:, 6:, :]
                pred_head = F.normalize(pred_head, dim=1)            
                head_ang = torch.mean(acos_safe(torch.sum(gt_head*pred_head, 1)))/torch.tensor(math.pi) * 180.0
                
                # head prediction error           
                gt_head_future = F.normalize(gt_head_future, dim=1)
                pred_head_future = F.normalize(pred_head_future, dim=1)
                baseline_head_future = gt_head[:, :, -1:].expand(-1, -1, self.conf.seq_len_future).clone()
                head_ang_future = torch.mean(acos_safe(torch.sum(gt_head_future*pred_head_future, 1)))/torch.tensor(math.pi) * 180.0
                head_ang_future_baseline = torch.mean(acos_safe(torch.sum(gt_head_future*baseline_head_future, 1)))/torch.tensor(math.pi) * 180.0

                self.log("val_hand_traj", hand_traj, on_epoch=True, prog_bar=True)
                self.log("val_head_ang", head_ang, on_epoch=True, prog_bar=True)
                self.log("val_hand_traj_future", hand_traj_future, on_epoch=True, prog_bar=True)
                self.log("val_head_ang_future", head_ang_future, on_epoch=True, prog_bar=True)
                self.log("val_hand_traj_future_baseline", hand_traj_future_baseline, on_epoch=True, prog_bar=True)
                self.log("val_head_ang_future_baseline", head_ang_future_baseline, on_epoch=True, prog_bar=True)

        if self.conf.in_channels == 6: # hand only
            if((self.current_epoch+1)% self.eval_every_epochs == 0):
                batch_future = batch[:, :, self.conf.seq_len:]
                gt_hand_future = batch_future[:, :, :]                
                batch = batch[:, :, :self.conf.seq_len]
                cond, pred_hand_future, pred_head_future = self.encode(batch)
                xT = self.encode_stochastic(batch, cond)
                pred_xstart = self.generate(xT, cond)
                
                # hand reconstruction error
                gt_hand = batch[:, :, :]
                pred_hand = pred_xstart[:, :, :]                        
                bs, channels, seq_len = gt_hand.shape
                gt_hand = gt_hand.reshape(bs, 2, 3, seq_len)
                pred_hand = pred_hand.reshape(bs, 2, 3, seq_len)            
                hand_traj = torch.mean(torch.norm(gt_hand - pred_hand, dim=2))
                
                # hand prediction error
                bs, channels, seq_len = gt_hand_future.shape
                gt_hand_future = gt_hand_future.reshape(bs, 2, 3, seq_len)            
                pred_hand_future = pred_hand_future.reshape(bs, 2, 3, seq_len)
                baseline_hand_future = gt_hand[:, :, :, -1:].expand(-1, -1, -1, self.conf.seq_len_future).clone()
                hand_traj_future = torch.mean(torch.norm(gt_hand_future - pred_hand_future, dim=2))
                hand_traj_future_baseline = torch.mean(torch.norm(gt_hand_future - baseline_hand_future, dim=2))
                
                self.log("val_hand_traj", hand_traj, on_epoch=True, prog_bar=True)
                self.log("val_hand_traj_future", hand_traj_future, on_epoch=True, prog_bar=True)
                self.log("val_hand_traj_future_baseline", hand_traj_future_baseline, on_epoch=True, prog_bar=True)

        if self.conf.in_channels == 3: # head only
            if((self.current_epoch+1)% self.eval_every_epochs == 0):
                batch_future = batch[:, :, self.conf.seq_len:]
                gt_head_future = batch_future[:, :, :]
                batch = batch[:, :, :self.conf.seq_len]
                cond, pred_hand_future, pred_head_future = self.encode(batch)
                xT = self.encode_stochastic(batch, cond)
                pred_xstart = self.generate(xT, cond)
                                               
                # head reconstruction error
                gt_head = batch[:, :, :]
                gt_head = F.normalize(gt_head, dim=1) # normalize head orientation to unit vectors
                pred_head = pred_xstart[:, :, :]
                pred_head = F.normalize(pred_head, dim=1)
                head_ang = torch.mean(acos_safe(torch.sum(gt_head*pred_head, 1)))/torch.tensor(math.pi) * 180.0
                
                # head prediction error           
                gt_head_future = F.normalize(gt_head_future, dim=1)
                pred_head_future = F.normalize(pred_head_future, dim=1)
                baseline_head_future = gt_head[:, :, -1:].expand(-1, -1, self.conf.seq_len_future).clone()
                head_ang_future = torch.mean(acos_safe(torch.sum(gt_head_future*pred_head_future, 1)))/torch.tensor(math.pi) * 180.0
                head_ang_future_baseline = torch.mean(acos_safe(torch.sum(gt_head_future*baseline_head_future, 1)))/torch.tensor(math.pi) * 180.0

                self.log("val_head_ang", head_ang, on_epoch=True, prog_bar=True)
                self.log("val_head_ang_future", head_ang_future, on_epoch=True, prog_bar=True)
                self.log("val_head_ang_future_baseline", head_ang_future_baseline, on_epoch=True, prog_bar=True)

            
    def test_step(self, batch, batch_idx):
        batch_future = batch[:, :, self.conf.seq_len:]
        gt_hand_future = batch_future[:, :6, :]
        gt_head_future = batch_future[:, 6:, :]
        batch = batch[:, :, :self.conf.seq_len]
        cond, pred_hand_future, pred_head_future = self.encode(batch)
        xT = self.encode_stochastic(batch, cond)
        pred_xstart = self.generate(xT, cond)
        
        # hand reconstruction error
        gt_hand = batch[:, :6, :]
        pred_hand = pred_xstart[:, :6, :]                        
        bs, channels, seq_len = gt_hand.shape
        gt_hand = gt_hand.reshape(bs, 2, 3, seq_len)
        pred_hand = pred_hand.reshape(bs, 2, 3, seq_len)            
        hand_traj = torch.mean(torch.norm(gt_hand - pred_hand, dim=2))
        
        # hand prediction error
        bs, channels, seq_len = gt_hand_future.shape
        gt_hand_future = gt_hand_future.reshape(bs, 2, 3, seq_len)            
        pred_hand_future = pred_hand_future.reshape(bs, 2, 3, seq_len)
        baseline_hand_future = gt_hand[:, :, :, -1:].expand(-1, -1, -1, self.conf.seq_len_future).clone()
        hand_traj_future = torch.mean(torch.norm(gt_hand_future - pred_hand_future, dim=2))
        hand_traj_future_baseline = torch.mean(torch.norm(gt_hand_future - baseline_hand_future, dim=2))
        
        # head reconstruction error
        gt_head = batch[:, 6:, :]
        gt_head = F.normalize(gt_head, dim=1) # normalize head orientation to unit vectors
        pred_head = pred_xstart[:, 6:, :]
        pred_head = F.normalize(pred_head, dim=1)            
        head_ang = torch.mean(acos_safe(torch.sum(gt_head*pred_head, 1)))/torch.tensor(math.pi) * 180.0
        
        # head prediction error           
        gt_head_future = F.normalize(gt_head_future, dim=1)
        pred_head_future = F.normalize(pred_head_future, dim=1)
        baseline_head_future = gt_head[:, :, -1:].expand(-1, -1, self.conf.seq_len_future).clone()
        head_ang_future = torch.mean(acos_safe(torch.sum(gt_head_future*pred_head_future, 1)))/torch.tensor(math.pi) * 180.0
        head_ang_future_baseline = torch.mean(acos_safe(torch.sum(gt_head_future*baseline_head_future, 1)))/torch.tensor(math.pi) * 180.0

        self.log("test_hand_traj", hand_traj, on_epoch=True, prog_bar=True)
        self.log("test_head_ang", head_ang, on_epoch=True, prog_bar=True)                    
        self.log("test_hand_traj_future", hand_traj_future, on_epoch=True, prog_bar=True)
        self.log("test_head_ang_future", head_ang_future, on_epoch=True, prog_bar=True)        
        self.log("test_hand_traj_future_baseline", hand_traj_future_baseline, on_epoch=True, prog_bar=True)            
        self.log("test_head_ang_future_baseline", head_ang_future_baseline, on_epoch=True, prog_bar=True)
       
        
    def generate(self, noise, cond=None, ema=True, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            
        if ema:
            model = self.ema_model
        else:
            model = self.model
                                  
        gen = sampler.sample(model=model, noise=noise, model_kwargs={'cond': cond})
        return gen
        
    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            
            ema(self.model, self.ema_model, self.conf.ema_decay)
            
            if (batch_idx==len(self.train_dataloader())-1) and ((self.current_epoch+1) % self.save_every_epochs == 0):
                save_path = os.path.join(self.conf.logdir, 'epoch_%d.ckpt' % (self.current_epoch+1))
                torch.save({
                    'state_dict': self.state_dict(),
                    'global_step': self.global_step,
                    'loss': outputs['loss'],
                }, save_path)
                
    def on_before_optimizer_step(self, optimizer: Optimizer,
                                 optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0: 
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out
        
        
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup

        
def train(conf: TrainConfig, model: LitModel, gpus):
    checkpoint = ModelCheckpoint(dirpath=conf.logdir,
                                 filename='last',
                                 save_last=True,
                                 save_top_k=1,
                                 every_n_epochs=conf.save_every_epochs,
                                 )
    checkpoint_path = f'{conf.logdir}last.ckpt'
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        if conf.mode == 'train':
            print('ckpt path:', checkpoint_path)
    else:
        print('checkpoint not found!')
        resume = None
            
    wandb_logger = pl_loggers.WandbLogger(project='haheae', 
                        name='%s_%s'%(model.conf.data_name, conf.logdir.split('/')[-2]),
                        log_model=True,
                        save_dir=conf.logdir, 
                        dir = conf.logdir,
                        config=vars(model.conf), 
                        save_code=True,
                        settings=wandb.Settings(code_dir="."))
                        
    trainer = pl.Trainer(
        max_epochs=conf.total_epochs,
        resume_from_checkpoint=resume,
        gpus=gpus,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        logger= wandb_logger,
        accumulate_grad_batches=conf.accum_batches,
        progress_bar_refresh_rate=4,
    )
    
    if conf.mode == 'train':
        trainer.fit(model)
    elif conf.mode == 'eval':
        checkpoint_path = f'{conf.logdir}last.ckpt'
        # load the latest checkpoint
        print('loading from:', checkpoint_path)
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        
        test_datasets = ['egobody', 'adt', 'gimo']
        for dataset_name in test_datasets:
            if dataset_name == 'egobody':
                data_dir = "/scratch/hu/pose_forecast/egobody_pose2gaze/"
                test_data = load_egobody(data_dir, conf.seq_len+conf.seq_len_future, 1, train=0) # use the test set
            elif dataset_name == 'adt':
                data_dir = "/scratch/hu/pose_forecast/adt_pose2gaze/"
                test_data = load_adt(data_dir, conf.seq_len+conf.seq_len_future, 1, train=2) # use the train+test set
            elif dataset_name == 'gimo':
                data_dir = "/scratch/hu/pose_forecast/gimo_pose2gaze/"
                test_data = load_gimo(data_dir, conf.seq_len+conf.seq_len_future, 1, train=2) # use the train+test set
                
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=conf.batch_size_eval, shuffle=False)        
            results = trainer.test(model, dataloaders=test_dataloader, verbose=False)
            print("\n\nTest on {}, dataset size: {}".format(dataset_name, test_data.shape))            
            print("test_hand_traj: {:.3f} cm".format(results[0]['test_hand_traj']*100))
            print("test_head_ang: {:.3f} deg".format(results[0]['test_head_ang']))
            print("test_hand_traj_future: {:.3f} cm".format(results[0]['test_hand_traj_future']*100))
            print("test_head_ang_future: {:.3f} deg".format(results[0]['test_head_ang_future']))
            print("test_hand_traj_future_baseline: {:.3f} cm".format(results[0]['test_hand_traj_future_baseline']*100))
            print("test_head_ang_future_baseline: {:.3f} deg\n\n".format(results[0]['test_head_ang_future_baseline']))
            
    wandb.finish()

    
def acos_safe(x, eps=1e-6):
    slope = np.arccos(1-eps) / eps
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf

    
def get_representation(model, dataset, conf, device='cuda'):
    model = model.to(device)
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size_eval, shuffle=False)
    with torch.no_grad():
        conds = [] # semantic representation
        xTs = [] # stochastic representation
        for batch in tqdm(dataloader, total=len(dataloader), desc='infer'):
            batch = batch.to(device)
            cond, _, _ = model.encode(batch)
            xT = model.encode_stochastic(batch, cond)            
            cond_cpu = cond.cpu().data.numpy()
            xT_cpu = xT.cpu().data.numpy()
            if len(conds) == 0:
                conds = cond_cpu
                xTs = xT_cpu
            else:
                conds = np.concatenate((conds, cond_cpu), axis=0)
                xTs = np.concatenate((xTs, xT_cpu), axis=0)                
    return conds, xTs

    
def generate_from_representation(model, conds, xTs, device='cuda'):
    model = model.to(device)
    model.eval()
    conds = torch.from_numpy(conds).to(device)
    xTs = torch.from_numpy(xTs).to(device)
    rec = model.generate(xTs, conds)
    rec = rec.cpu().data.numpy()    
    return rec


def evaluate_reconstruction(gt, rec):
    # hand reconstruction error (cm)
    gt_hand = gt[:, :6, :]
    rec_hand = rec[:, :6, :]                        
    bs, channels, seq_len = gt_hand.shape
    gt_hand = gt_hand.reshape(bs, 2, 3, seq_len)
    rec_hand = rec_hand.reshape(bs, 2, 3, seq_len)            
    hand_traj_errors = np.mean(np.mean(np.linalg.norm(gt_hand - rec_hand, axis=2), axis=1), axis=1)*100
    
    # head reconstruction error (deg)
    gt_head = gt[:, 6:, :]
    gt_head_norm = np.linalg.norm(gt_head, axis=1, keepdims=True)
    gt_head = gt_head/gt_head_norm
    rec_head = rec[:, 6:, :]
    rec_head_norm = np.linalg.norm(rec_head, axis=1, keepdims=True)
    rec_head = rec_head/rec_head_norm
    dot_sum = np.clip(np.sum(gt_head*rec_head, axis=1), -1, 1)
    head_ang_errors = np.mean(np.arccos(dot_sum), axis=1)/np.pi * 180.0
    return hand_traj_errors, head_ang_errors

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=7, type=int)
    parser.add_argument('--mode', default='eval', type=str)
    parser.add_argument('--encoder_type', default='gcn', type=str)
    parser.add_argument('--model_name', default='haheae', type=str)
    parser.add_argument('--hand_mse_factor', default=1.0, type=float)
    parser.add_argument('--head_mse_factor', default=1.0, type=float)
    parser.add_argument('--data_sample_rate', default=1, type=int)
    parser.add_argument('--epoch', default=130, type=int)
    parser.add_argument('--in_channels', default=9, type=int)    
    args = parser.parse_args()
    
    conf = egobody_autoenc(args.mode, args.encoder_type, args.hand_mse_factor, args.head_mse_factor, args.data_sample_rate, args.epoch, args.in_channels)
    model = LitModel(conf)
    conf.logdir = f'{conf.logdir}{args.model_name}/'
    print('log dir: {}'.format(conf.logdir))
    MakeDir(conf.logdir)
    
    if conf.mode == 'train' or conf.mode == 'eval': # train or evaluate the model
        os.environ['WANDB_CACHE_DIR'] = conf.logdir
        os.environ['WANDB_DATA_DIR'] = conf.logdir
        # set wandb to not upload checkpoints, but all the others
        os.environ['WANDB_IGNORE_GLOBS'] = '*.ckpt'
        local_time = time.asctime(time.localtime(time.time()))
        print('\n{} starts at {}'.format(conf.mode, local_time))
        start_time = datetime.datetime.now()
        train(conf, model, gpus=[args.gpus])
        end_time = datetime.datetime.now()
        total_time = (end_time - start_time).seconds/60
        print('\nTotal time: {:.3f} min'.format(total_time))
        local_time = time.asctime(time.localtime(time.time()))
        print('\n{} ends at {}'.format(conf.mode, local_time))