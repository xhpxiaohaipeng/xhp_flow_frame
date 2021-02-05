# -*- coding: utf-8 -*-
from io import open
import os.path
from os import path
import random
import numpy as np
import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from glob import glob
file_path = np.array(glob('data/train/*'))
import seaborn as sns
class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length=40, predict_length=30, file_path=file_path):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """

        self.X_frames_trajectory = []
        self.Y_frames_trajectory = []
        self.length = length
        self.predict_length = predict_length
        for csv_file in file_path:
            self.csv_file = csv_file
            self.load_data()
        #self.normalize_data()

    def __len__(self):
        return len(self.X_frames_trajectory)

    def __getitem__(self, idx):
        single_trajectory_data = self.X_frames_trajectory[idx]
        single_trajectory_label = self.Y_frames_trajectory[idx]
        return (single_trajectory_data, single_trajectory_label)

    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        dataS = dataS.sort_values(by="time_us", ascending=True)
        #dataS = dataS[dataS.Class == 5]
        #dataS = dataS[dataS.Static == 0]
        #dataS = dataS[dataS.Label != 7]
        #dataS = dataS[dataS.ID != -1]
        count_ = []
        frame = dataS[
            ["speed_ms","acceleration",	"yaw","steer_angle","throttle_status","break_status","auto_state"]]
       # sns.heatmap(frame.corr(), annot=True, fmt='.1f')  #查看相关性
       # plt.show()
        #plt.savefig("相关性点.png")
        total_frame_data = np.asarray(frame)
        X = total_frame_data[:-self.predict_length, :]  # 预测1个轨迹点
        Y = total_frame_data[self.predict_length:, :1]
        #             print(X.shape,Y.shape)

        count = 0
        for i in range(X.shape[0] - self.length):
            # if random.random() > 0.2:   #-------------------------------
            # continue
            #                 if count > 60:  # 限制每辆车的最大轨迹数
            #                      break
            #  print('X[] shape',X[i:i+100,:].shape)

            self.X_frames_trajectory = self.X_frames_trajectory + [
                X[i:i + self.length, :]]  # 生成轨迹段,每个轨迹为100个点,所有轨迹集合,组合成输入数据
            self.Y_frames_trajectory = self.Y_frames_trajectory + [Y[i:i + self.length, :]]  # 生成对应的label
            count = count + 1
        count_.append(count)
        print('File:', self.csv_file.split("/")[2], " Total trajectory point:",
              total_frame_data.shape[0], 'Total Trajectory:', count)


        # print(np.array(self.X_frames_trajectory).shape,np.array(self.Y_frames_trajectory).shape)

    def normalize_data(self):  # 标准化每辆车的输入数据
        # 输出轨迹预测数据，进行标准化
        A = [list(x) for x in zip(*(self.X_frames_trajectory))]
        A = np.array(A).astype(np.float64)
        # A = torch.tensor(A)
        A = torch.from_numpy(A)
        print(A.shape)
        A = A.view(-1, A.shape[2])
        print('A shape:', A.shape)
        if self.csv_file.split("/")[1] == 'train':
            self.mn = torch.mean(A, dim=0)
            # print(self.mn.shape)
            self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values) / 2.0
            self.range = torch.ones(self.range.shape, dtype=torch.double)
            # print(self.range.shape)
            self.std = torch.std(A, dim=0)
            #   print(self.std.shape)
            std = self.std.numpy()
            mn = self.mn.numpy()
            rg = self.range.numpy()
            np.savetxt("txt/std.txt", std)
            np.savetxt("txt/mean.txt", mn)
            np.savetxt("txt/rg.txt", rg)
        else:
            mn= torch.from_numpy(np.loadtxt('txt/mean.txt'))
            std = torch.from_numpy(np.loadtxt('txt/std.txt'))
            rg = torch.from_numpy(np.loadtxt('txt/rg.txt'))
            self.mn = mn
            self.range = rg
            self.std = std
        self.X_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn) / (self.std * self.range) for item in
            self.X_frames_trajectory]
        self.Y_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn[:1]) / (self.std[:1] * self.range[:1]) for
            item in self.Y_frames_trajectory]


def get_dataloader(BatchSize=64, length=10, predict_length=1,file_path = np.array(glob('data/train/*')),daset = 'train'):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    # load dataset
    if path.exists("pickle/dataset_traj_{}_1221_7_1_{}_{}.pickle".format(daset,predict_length, length)):
        with open('pickle/dataset_traj_{}_1221_7_1_{}_{}.pickle'.format(daset,predict_length, length), 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = TrajectoryDataset(length, predict_length,file_path)
        with open('pickle/dataset_traj_{}_1221_7_1_{}_{}.pickle'.format(daset,predict_length, length), 'wb') as output:
            pickle.dump(dataset, output)
    # split dataset into train test and validation 8:1:1
    length_traj = dataset.__len__()
    #num_train_traj = (int)(length_traj * 0.8)
   # num_test_traj = (int)(length_traj * 0.9) - num_train_traj
    #num_validation_traj = (int)(length_traj - num_test_traj - num_train_traj)

   # train_traj, test_traj, validation_traj = torch.utils.data.random_split(dataset, [num_train_traj, num_test_traj,
                                                                                    # num_validation_traj])

    train_loader_traj = DataLoader(dataset, batch_size=BatchSize, shuffle=True)
    #test_loader_traj = DataLoader(test_traj, batch_size=BatchSize, shuffle=True)
   # validation_loader_traj = DataLoader(validation_traj, batch_size=BatchSize, shuffle=True)
    iters = iter(train_loader_traj)
    x_trajectory, y_trajectory = next(iters)
    print("*" * 100)
    if daset == 'train':
        print('训练轨迹轨迹条数：', length_traj)
    if daset == 'valid':
        print('验证轨迹轨迹条数：', length_traj)
    if daset == 'test':
        print('测试轨迹轨迹条数：', length_traj)
    print('---轨迹输入数据结构：', x_trajectory.shape, '---轨迹输出数据结构：', y_trajectory.shape)
    print('---轨迹长度：', length, '---预测轨迹长度：', predict_length)
    return (train_loader_traj, dataset)
