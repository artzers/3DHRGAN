
import os, tifffile
import Net, torch
import numpy as np
from models import WDSRB
from libtiff import TIFFimage, TIFF
import time
from tqdm import tqdm

img = tifffile.imread('D:/work_note/DualSR3D/brainregion/lowCrop.tif')
img = img[:160,:660,:660]

minLowRange=[0,0,0]#[1800,3300,550]#z,y,x
minLowRange = minLowRange[-1::-1]
maxLowRange=[660,660,160]#[2500,4000,750]
maxLowRange = maxLowRange[-1::-1]#reverse
readRange = [60, 60, 60]
zMinLowList = []
zMaxLowList = []
yMinLowList = []
yMaxLowList = []
xMinLowList = []
xMaxLowList = []

for k in range(minLowRange[0], maxLowRange[0] - 39, 20):
    zMinLowList.append(k)
    zMaxLowList.append(k+40)

for k in range(minLowRange[1], maxLowRange[1] - 59,40):
    #for k in range(minLowRange[1], maxLowRange[1] - 40,20):
    yMinLowList.append(k)
    yMaxLowList.append(k+60)#59

for k in range(minLowRange[2], maxLowRange[2] - 59,40):
    xMinLowList.append(k)
    xMaxLowList.append(k+60)

#
#pretrained_net = torch.load('D:/Python/SR201905/20190528-DualGAN-5X20X/saved_models/G_AB_36000.pth')
pretrained_net = WDSRB()
#pretrained_net.load_state_dict(torch.load('D:/Python/SR201905/20190528-DualGAN-5X20X/G_AB_91000.pth'))
pretrained_net.load_state_dict(torch.load('D:/Python/SR201905/20190528-DualGAN-5X20X/saved_models_Soma/G_AB_20000.pth'))
pretrained_net = pretrained_net.cuda(0)
pretrained_net.eval()
torch.set_grad_enabled(False)
torch.cuda.empty_cache()


lowMeanVal = 600
lowStdVal = 400
highMeanVal = 2500
highStdVal = 2000
highImg = np.zeros((np.array(maxLowRange) - np.array(minLowRange))*3,dtype=np.uint16)
xBase = xMinLowList[0]
yBase = yMinLowList[0]
zBase = zMinLowList[0]
time_start = time.time()
for i in range(len(zMinLowList)):#TODO
    for j in range(len(yMinLowList)):
        for k in range(len(xMinLowList)):
            print('processing %d-%d, %d-%d %d-%d'%(xMinLowList[k], xMaxLowList[k],
                                                   yMinLowList[j], yMaxLowList[j],
                                                   zMinLowList[i], zMaxLowList[i]))
            # lowImg = reader.SelectIOR(xMinLowList[k], xMaxLowList[k],
            #                           yMinLowList[j], yMaxLowList[j],
            #                           zMinLowList[i], zMaxLowList[i],1)
            #print('Cache : %d'%len(reader.imageCacheManager.cacheList))
            lowImg = img[zMinLowList[i]: zMaxLowList[i],
                     yMinLowList[j]: yMaxLowList[j],
                     xMinLowList[k]: xMaxLowList[k]]

            lowImg = np.array(lowImg, dtype=np.float32)
            lowImg = (lowImg - lowMeanVal) / (lowStdVal)
            lowImg = np.expand_dims(lowImg, axis=0)
            lowImg = np.expand_dims(lowImg, axis=0)
            lowImg = torch.from_numpy(lowImg).float()
            lowImg = lowImg.cuda(1)
            pre2 = pretrained_net(lowImg)
            saveImg = pre2.cpu().data.numpy()[0, 0, :, :, :]
            saveImg *= lowStdVal
            saveImg += lowMeanVal
            saveImg = np.uint16(np.maximum(np.minimum(saveImg, 65535), 0))


            highImg[(zMinLowList[i]-zBase)*3+30:(zMinLowList[i]-zBase)*3+90,
            (yMinLowList[j]-yBase)*3+30:(yMinLowList[j]-yBase)*3+150,
            (xMinLowList[k]-xBase)*3+30:(xMinLowList[k]-xBase)*3+150] = saveImg[30:90, 30:150, 30:150]

time_end = time.time()
print('totally time cost', time_end - time_start)
tifffile.imwrite('D:/work_note/DualSR3D/brainregion//highcrop20191118.tif', highImg[60:-60,60:-60,60:-60])
