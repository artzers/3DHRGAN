import os, torch, tifffile
import numpy as np
from torch import nn
from torch.nn import functional as F
from vis import vis_tool
import tifffile
import cv2
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs
import torch.nn.utils as utils
from torch.utils.checkpoint import checkpoint
#from gpu_mem_track import  MemTracker
import inspect
import math, cv2
from tqdm import tnrange, tqdm_notebook
from scipy.ndimage.interpolation import zoom
from scipy.misc import imresize
from skimage.transform import resize

#skimage.measure.compare_psnr(im_true, im_test, data_range=None, dynamic_range=None)
#skimage.measure.compare_ssim(


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def prepare(dev, *args):
    # print(dev)
    device = torch.device(dev)
    if dev == 'cpu':
        device = torch.device('cpu')
    return [a.to(device) for a in args]


def calc_psnr(sr, hr, scale):
    diff = (sr - hr)
    # shave = scale + 6
    # valid = diff[..., shave:-shave, shave:-shave,:]#2，2，1
    # mse = valid.pow(2).mean()
    mse = np.mean(diff * diff) + 0.0001
    return -10 * np.log10(mse / (4095 ** 2))


def RestoreNetImg(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    maxVal = np.max(rImg)
    minVal = np.min(rImg)
    rImg = 255./(maxVal - minVal+1) * (rImg - minVal)
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

def RestoreNetImgV2(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

class WDSRBBlock3D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBBlock3D, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv3d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv3d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv3d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # res = self.body(x) * self.res_scale
        # res += x
        res = self.body(x) + x
        return res

class ResBlock3D(nn.Module):
    def __init__(self,
                 conv=default_conv3d,
                 n_feats=64,
                 kernel_size=3,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ResBlock3D, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class PixelUpsampler3D(nn.Module):
    def __init__(self,
                 upscaleFactor,
                 # conv=default_conv3d,
                 # n_feats=32,
                 # kernel_size=3,
                 # bias=True
                 ):
        super(PixelUpsampler3D, self).__init__()
        self.scaleFactor = upscaleFactor

    def PixelShuffle(self, input, upscaleFactor):
        batchSize, channels, inDepth, inHeight, inWidth = input.size()
        channels //= upscaleFactor[0] * upscaleFactor[1] * upscaleFactor[2]
        outDepth = inDepth * upscaleFactor[0]
        outHeight = inHeight * upscaleFactor[1]
        outWidth = inWidth * upscaleFactor[2]
        inputView = input.contiguous().view(
            batchSize, channels, upscaleFactor[0], upscaleFactor[1], upscaleFactor[2], inDepth,
            inHeight, inWidth)
        shuffleOut = inputView.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffleOut.view(batchSize, channels, outDepth, outHeight, outWidth)

    def forward(self, x):
        # x = self.conv(x)
        up = self.PixelShuffle(x, self.scaleFactor)
        return up




class GetTrainDataSet2():
    def __init__(self, lrDir, hrDir ):
        self.lrDir = lrDir
        self.hrDir = hrDir
        self.lrFileList = []
        self.hrFileList = []
        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return len(self.hrFileList)

    def __getitem__(self, ind):
        # load the image and labels
        imgName = self.hrFileList[ind]
        lrName = os.path.join(self.lrDir, imgName)
        hrName = os.path.join(self.hrDir, imgName)

        lrImg = tifffile.imread(lrName)
        hrImg = tifffile.imread(hrName)
        #lrImg = lrImg[:32,:32,:32]
        #hrImg = hrImg[:96, :96, :96]
        # print(lrImg.shape)
        # print(hrImg.shape)

        # randX = np.random.randint(0, 100-32 - 1)
        # randY = np.random.randint(0, 100 - 32 -1 )
        # #randZ = np.random.randint(0, 30 - 24 - 1)
        # lrImg = lrImg[:, randY:randY+32, randX:randX+32]#randZ:randZ+24
        # hrImg = hrImg[:,#randZ*3:randZ*3 + 72,
        #         randY*3:randY*3 + 96,
        #         randX*3:randX*3 + 96]
        # print(lrImg.shape)
        # print(hrImg.shape)
        lrImg = np.expand_dims(lrImg, axis=0)
        hrImg = np.expand_dims(hrImg, axis=0)

        lrImg = np.array(lrImg, dtype=np.float32)
        hrImg = np.array(hrImg, dtype=np.float32)

        lrImg = torch.from_numpy(lrImg).float()
        hrImg = torch.from_numpy(hrImg).float()
        return lrImg, hrImg

class GetSimTrainDataSet2():
    def __init__(self, hrDir, mean, std):
        self.hrDir = hrDir
        self.hrFileList = []
        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        # self.mean1=np.array([160],dtype=np.float32)
        self.mean1 = mean  # np.array([127], dtype=np.float32)
        self.std1 = std  # np.array([350], dtype=np.float32)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return len(self.hrFileList)

    def __getitem__(self, ind):
        # load the image and labels
        imgName = self.hrFileList[ind]
        hrName = os.path.join(self.hrDir, imgName)

        hrImg = tifffile.imread(hrName)
        lrImg = hrImg.copy()
        for k in range(lrImg.shape[0]):
            lrImg[k,:,:] = cv2.blur(lrImg[k,:,:],(5,5))
        lrImg = zoom(lrImg,1./3.)

        # print(lrImg.shape)
        # print(hrImg.shape)

        randX = np.random.randint(10, 100-48 - 1)
        randY = np.random.randint(10, 100 - 48 -1 )
        randZ = np.random.randint(10, 30 - 16 - 1)
        lrImg = lrImg[randZ:randZ + 32, randY:randY+32, randX:randX+32]
        hrImg = hrImg[randZ*3:randZ*3 + 96,
                randY*3:randY*3 + 96,
                randX*3:randX*3 + 96]
        # print(lrImg.shape)
        # print(hrImg.shape)

        lrImg = np.array(lrImg, dtype=np.float32)
        hrImg = np.array(hrImg, dtype=np.float32)

        lrImg = (lrImg - self.mean1) / self.std1
        hrImg = (hrImg - self.mean1) / self.std1

        lrImg = np.expand_dims(lrImg, axis=0)
        hrImg = np.expand_dims(hrImg, axis=0)

        # torch.set_grad_enabled(True)
        lrImg = torch.from_numpy(lrImg).float()
        hrImg = torch.from_numpy(hrImg).float()
        return lrImg, hrImg


class GetTestDataSet():
    def __init__(self, testDir, mean, std):
        self.testDir = testDir
        self.testFileList = os.listdir(self.testDir)

        # self.mean1=np.array([160],dtype=np.float32)
        self.mean1 = mean  # np.array([127], dtype=np.float32)
        self.std1 = std  # np.array([350], dtype=np.float32)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.testFileList)

    def __len__(self):
        return len(self.testFileList)

    def __getitem__(self, ind):
        # load the image and labels
        imgName = self.testFileList[ind]
        lrName = os.path.join(self.testDir, imgName)
        lrImg = tifffile.imread(lrName)

        lrImg = np.array(lrImg, dtype=np.float32)
        lrImg = (lrImg - self.mean1) / self.std1
        lrImg = np.expand_dims(lrImg, axis=0)
        # torch.set_grad_enabled(True)
        lrImg = torch.from_numpy(lrImg).float()
        return lrImg
