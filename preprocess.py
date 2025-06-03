import os, random, math, copy
import pandas as pd
import numpy as np
import pickle as pkl
import logging, sys
from torch.utils.data import DataLoader,Dataset
import multiprocessing as mp
import json
import matplotlib.pyplot as plt


def MakeDir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

        
def load_egobody(data_dir, seq_len, sample_rate=1, train=1):
        data_dir_train = data_dir + 'train/'
        data_dir_test = data_dir + 'test/'
        if train == 0:
            data_dirs = [data_dir_test] # test
        elif train == 1:
            data_dirs = [data_dir_train] # train
        elif train == 2:
            data_dirs = [data_dir_train, data_dir_test] # train + test
        
        hand_head = []
        for data_dir in data_dirs:
            file_paths = sorted(os.listdir(data_dir))
            pose_xyz_file_paths = []
            head_file_paths = []
            for path in file_paths:
                path_split = path.split('_')
                data_type = path_split[-1][:-4]
                if(data_type == 'xyz'):
                    pose_xyz_file_paths.append(path)
                if(data_type == 'head'):
                    head_file_paths.append(path)    
                
            file_num = len(pose_xyz_file_paths)        
            for i in range(file_num):
                pose_data = np.load(data_dir + pose_xyz_file_paths[i])
                head_data = np.load(data_dir + head_file_paths[i])
                num_frames = pose_data.shape[0]
                if num_frames < seq_len:
                    continue
                
                head_pos = pose_data[:, 15*3:16*3]
                left_hand_pos = pose_data[:, 20*3:21*3]
                right_hand_pos = pose_data[:, 21*3:22*3]
                head_ori = head_data
                left_hand_pos -= head_pos # convert hand positions to head coordinate system
                right_hand_pos -= head_pos
                hand_head_data = left_hand_pos
                hand_head_data = np.concatenate((hand_head_data, right_hand_pos), axis=1)
                hand_head_data = np.concatenate((hand_head_data, head_ori), axis=1)
                            
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = hand_head_data[fs_sel, :]
                seq_sel = seq_sel[0::sample_rate, :, :]                    
                if len(hand_head) == 0:
                    hand_head = seq_sel
                else:
                    hand_head = np.concatenate((hand_head, seq_sel), axis=0)
        
        hand_head = np.transpose(hand_head, (0, 2, 1))
        return hand_head

        
def load_adt(data_dir, seq_len, sample_rate=1, train=1):
        data_dir_train = data_dir + 'train/'
        data_dir_test = data_dir + 'test/'
        if train == 0:
            data_dirs = [data_dir_test] # test
        elif train == 1:
            data_dirs = [data_dir_train] # train
        elif train == 2:
            data_dirs = [data_dir_train, data_dir_test] # train + test
            
        hand_head = []
        for data_dir in data_dirs:
            file_paths = sorted(os.listdir(data_dir))
            pose_xyz_file_paths = []
            head_file_paths = []
            for path in file_paths:
                path_split = path.split('_')
                data_type = path_split[-1][:-4]
                if(data_type == 'xyz'):
                    pose_xyz_file_paths.append(path)
                if(data_type == 'head'):
                    head_file_paths.append(path)    
                
            file_num = len(pose_xyz_file_paths)        
            for i in range(file_num):
                pose_data = np.load(data_dir + pose_xyz_file_paths[i])
                head_data = np.load(data_dir + head_file_paths[i])
                num_frames = pose_data.shape[0]
                if num_frames < seq_len:
                    continue
                
                head_pos = pose_data[:, 4*3:5*3]
                left_hand_pos = pose_data[:, 8*3:9*3]
                right_hand_pos = pose_data[:, 12*3:13*3]
                head_ori = head_data
                left_hand_pos -= head_pos # convert hand positions to head coordinate system
                right_hand_pos -= head_pos
                hand_head_data = left_hand_pos
                hand_head_data = np.concatenate((hand_head_data, right_hand_pos), axis=1)
                hand_head_data = np.concatenate((hand_head_data, head_ori), axis=1)
                            
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = hand_head_data[fs_sel, :]
                seq_sel = seq_sel[0::sample_rate, :, :]                    
                if len(hand_head) == 0:
                    hand_head = seq_sel
                else:
                    hand_head = np.concatenate((hand_head, seq_sel), axis=0)            
        
        hand_head = np.transpose(hand_head, (0, 2, 1))
        return hand_head
       
        
def load_gimo(data_dir, seq_len, sample_rate=1, train=1):
        data_dir_train = data_dir + 'train/'
        data_dir_test = data_dir + 'test/'
        if train == 0:
            data_dirs = [data_dir_test] # test
        elif train == 1:
            data_dirs = [data_dir_train] # train
        elif train == 2:
            data_dirs = [data_dir_train, data_dir_test] # train + test
            
        hand_head = []
        for data_dir in data_dirs:
            file_paths = sorted(os.listdir(data_dir))
            pose_xyz_file_paths = []
            head_file_paths = []
            for path in file_paths:
                path_split = path.split('_')
                data_type = path_split[-1][:-4]
                if(data_type == 'xyz'):
                    pose_xyz_file_paths.append(path)
                if(data_type == 'head'):
                    head_file_paths.append(path)    
            
            file_num = len(pose_xyz_file_paths)        
            for i in range(file_num):
                pose_data = np.load(data_dir + pose_xyz_file_paths[i])
                head_data = np.load(data_dir + head_file_paths[i])
                num_frames = pose_data.shape[0]
                if num_frames < seq_len:
                    continue
                
                head_pos = pose_data[:, 15*3:16*3]
                left_hand_pos = pose_data[:, 20*3:21*3]
                right_hand_pos = pose_data[:, 21*3:22*3]
                head_ori = head_data
                left_hand_pos -= head_pos # convert hand positions to head coordinate system
                right_hand_pos -= head_pos
                hand_head_data = left_hand_pos
                hand_head_data = np.concatenate((hand_head_data, right_hand_pos), axis=1)
                hand_head_data = np.concatenate((hand_head_data, head_ori), axis=1)
                            
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = hand_head_data[fs_sel, :]
                seq_sel = seq_sel[0::sample_rate, :, :]                    
                if len(hand_head) == 0:
                    hand_head = seq_sel
                else:
                    hand_head = np.concatenate((hand_head, seq_sel), axis=0)            
        
        hand_head = np.transpose(hand_head, (0, 2, 1))
        return hand_head

        
if __name__ == "__main__":
    data_dir = "/scratch/hu/pose_forecast/egobody_pose2gaze/"
    seq_len = 40
    
    test_data = load_egobody(data_dir, seq_len, sample_rate=10, train=0)
    print("\ndataset size: {}".format(test_data.shape))