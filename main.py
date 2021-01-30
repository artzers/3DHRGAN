import os, torch
import numpy as np
#import pynvml
from torch import nn
from torch.nn import functional as F
from vis import vis_tool
import tifffile
from libtiff import TIFFimage
import cv2
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs
import torch.nn.utils as utils
from torch.utils.checkpoint import checkpoint
#from gpu_mem_track import  MemTracker
import inspect
import math,time
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom
from Util import GetTestDataSet, GetSimTrainDataSet2, \
    GetTrainDataSet2,  \
PixelUpsampler3D, RestoreNetImg, calc_psnr, prepare
#from Net import Trainer
import Net
#import ParallelNet

'''
python -m visdom.server

http://localhost:8097/
'''

def CalcMeanStd(path):
    srcPath = path
    fileList = os.listdir(srcPath)
    fileNum = len(fileList)

    globalMean = 0
    globalStd = 0

    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        mean = np.mean(img)
        globalMean += mean
    globalMean /= fileNum


    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        img = img.astype(np.float)
        img -= globalMean
        sz = img.shape[0] * img.shape[1] * img.shape[2]
        globalStd += np.sum(img ** 2) / np.float(sz)
    globalStd = (globalStd / len(fileList)) ** (0.5)

    print(globalMean)
    print(globalStd)
    return globalMean,globalStd


def CalcMeanMax(path):
    srcPath = path
    fileList = os.listdir(srcPath)
    fileNum = len(fileList)

    maxVal = 0
    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        maxVal = np.maximum(maxVal, np.max(img))

    print(maxVal)

    globalMean = 0
    globalStd = 0

    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        mean = np.mean(img)
        globalMean += mean
    globalMean /= fileNum
    print(globalMean)
    return globalMean,maxVal


hrPath = 'D:/Document/wuhan202001/20190709Sample/20X/'
midPath = 'D:/Document/wuhan202001/20190709Sample/5X/'

env = 'DualGAN20190528'
globalDev = 'cuda:0'
globalDeviceID = 0

if __name__ == '__main__':
    lowMean,lowStd = 600,400 #CalcMeanStd(midPath)#
    highMean, highStd = 2500,2000#CalcMeanStd(hrPath)#
    #exit(0)#

    train_dataset = GetTrainDataSet2(midPath, hrPath)

    #train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    visdomable = True
    if visdomable == True:
        logger = vis_tool.Visualizer(env=env)
        logger.reinit(env=env)

    Net.logger = logger
    Net.lowMean = lowMean
    Net.lowStd = lowStd
    Net.highMean = highMean
    Net.highStd = highStd
    trainer = Net.TrainerDualWGANGP(data_loader=train_loader, test_loader=None)

    time_start = time.time()
    trainer.Train(turn=500)
    time_end = time.time()
    print('totally time cost', time_end - time_start)

