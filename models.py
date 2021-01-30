import torch.nn as nn
import torch.nn.functional as F
import torch
from Util import WDSRBBlock3D, PixelUpsampler3D


class WDSRB(nn.Module):
    def __init__(self):
        super(WDSRB, self).__init__()
        # hyper-params
        n_resblocks = 8
        n_feats = 32
        kernel_size = 3
        #upscaleFactor = [2, 2, 2]
        #blockFeatNum = 64

        act = nn.LeakyReLU(inplace=True)#nn.LeakyReLU(0.2,True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        head = []
        head.append(
            wn(nn.Conv3d(1, n_feats, 3, padding=3 // 2)))

        body = []
        for i in range(n_resblocks):
            body.append(
                #WDSRABlock3D(n_feats, kernel_size, blockFeatNum, wn=wn, act=act))
                WDSRBBlock3D(n_feats, kernel_size, wn=wn, act=act))


        out_feats = 3 * 3 * 3#3 * 3 * n_feats
        tail = []
        tail.append(
            wn(nn.Conv3d(n_feats, out_feats, 3, padding=3 // 2)))
        tail.append(PixelUpsampler3D((3, 3, 3)))

        #out_feats = 3 * n_feats
        # tailz = []
        # tailz.append(
        #     wn(nn.Conv3d(n_feats, out_feats, 3, padding=3 // 2)))
        # tailz.append(PixelUpsampler3D((3, 1, 1)))

        # skip
        #out_feats = 2 * 2 * n_feats
        skip = []
        skip.append(wn(nn.Conv3d(1, 3*3*3, 5, padding=5 // 2)))
        skip.append(PixelUpsampler3D((3, 3, 3)))
        #skip.append(wn(nn.Conv3d(1, out_feats, 5, padding=5 // 2)))
        #skip.append(PixelUpsampler3D((1, 2, 2)))
        #out_feats = 2 * n_feats
        #skip.append(wn(nn.Conv3d(n_feats, out_feats, 5, padding=5 // 2)))
        #skip.append(PixelUpsampler3D((2, 1, 1)))
        #self.skipconv = nn.Conv3d(1, 1, 5, padding=5 // 2)

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        # self.tailconv1 = nn.Conv3d(n_feats, n_feats, 3, padding=3 // 2)
        # self.tailconv2 = nn.Conv3d(n_feats, 1, 3, padding=3 // 2)
        #self.mean = torch.autograd.Variable(torch.FloatTensor(127.5)).view([1, 1, 1, 1])
        #self.std = torch.autograd.Variable(torch.FloatTensor(127.5)).view([1, 1, 1, 1])


    def forward(self, x):
        s = self.skip(x)
        #s = F.interpolate(x,None, (3,3,3),'trilinear',align_corners=True)
        #s = self.skipconv(s)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        return x

#downsample generator
class DegradeNet(nn.Module):
    def __init__(self):
        super(DegradeNet, self).__init__()
        self.nFeat = 64
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU#(inplace=True)

        layer1 = [
            nn.Conv3d(1,self.nFeat,3,3//2),
            act(inplace=True)
            ]
        layer2=[
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            #nn.MaxPool3d(3,stride = 3,padding=3//2),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3,  padding=3 // 2)),
            act(inplace=True),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=3, padding=3 // 2)),
            act(inplace=True),
            nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)
        ]

        self.head = nn.Sequential(*layer1)
        self.body = nn.Sequential(*layer2)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x


#discriminator low image
class LowDiscriminator(nn.Module):
    def __init__(self):
        super(LowDiscriminator, self).__init__()
        self.nFeat = 64
        self.output_shape = (1,1,1,1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU
        layer1 = [
            #wn(nn.Conv3d(1,self.nFeat,3,3//2)),
            (nn.Conv3d(1, self.nFeat, 3, 3 // 2)),
            act(inplace=True),
        ]
        layer2 = [
            #wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            nn.MaxPool3d(3, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            nn.MaxPool3d(3, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            nn.MaxPool3d(3, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)),
            #act(inplace=True),
            #nn.Conv3d(self.nFeat, 1, 2)
        ]
        self.head = nn.Sequential(*layer1)
        self.body = nn.Sequential(*layer2)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x

#discriminator high image
class HighDiscriminator(nn.Module):
    def __init__(self):
        super(HighDiscriminator, self).__init__()

        self.nFeat = 64
        self.output_shape = (1,1,1,1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU
        layer1 = [
            #wn(nn.Conv3d(1, self.nFeat, 3, padding=3 // 2)),
            (nn.Conv3d(1, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
        ]
        layer2 = [
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.MaxPool3d(2, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat,3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.MaxPool3d(2, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat,3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(True),
            nn.MaxPool3d(2, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat,3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.MaxPool3d(2, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat,3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            nn.Conv3d(self.nFeat,self.nFeat, 3, padding=3 // 2),
            act(inplace=True),
            nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2),
            act(inplace=True),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)
        ]
        self.head = nn.Sequential(*layer1)
        self.body = nn.Sequential(*layer2)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x

