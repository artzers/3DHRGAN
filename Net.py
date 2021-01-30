import os, torch, tifffile
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs
import torch.nn.utils as utils
from torch.utils.checkpoint import checkpoint

from scipy.ndimage.interpolation import zoom
import itertools
from torch.autograd import Variable
import torch.autograd as autograd
from models import WDSRB, DegradeNet, LowDiscriminator, HighDiscriminator

from Util import GetTestDataSet, GetTrainDataSet2, \
PixelUpsampler3D, RestoreNetImg, calc_psnr,  \
default_conv3d, prepare, RestoreNetImgV2, ResBlock3D

logger = None
lowMean =0
lowStd = 0
highMean =0
highStd = 0
globalMax = 0

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class TrainerDualWGANGP:
    def __init__(self,
                 data_loader,
                 test_loader,
                 scheduler=lrs.StepLR,
                 dev='cuda:0', devid=0):
        self.dataLoader = data_loader
        self.testLoader = test_loader
        # self.scheduler = scheduler(
        #     self.optimizer, step_size=2000, gamma=0.8, last_epoch=-1)
        self.dev = dev
        self.cudaid = devid

        # Loss function
        self.adversarial_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.L1Loss(reduce='mean')

        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_gp = 10

        # Initialize generator and discriminator
        self.G_AB = WDSRB()
        self.G_BA = DegradeNet()
        self.D_A = LowDiscriminator()
        self.D_B = HighDiscriminator()

        # self.G_AB.load_state_dict(torch.load('./saved_models20190614//G_AB_57000.pth'))
        # self.G_BA.load_state_dict(torch.load('./saved_models20190614//G_BA_57000.pth'))
        # self.D_A.load_state_dict(torch.load('./saved_models20190614//D_A_57000.pth'))
        # self.D_B.load_state_dict(torch.load('./saved_models20190614//D_B_57000.pth'))
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)

        self.G_AB.cuda(self.cudaid)
        self.G_BA.cuda(self.cudaid)
        self.D_A.cuda(self.cudaid)
        self.D_B.cuda(self.cudaid)
        self.cycle_loss.cuda(self.cudaid)
        # self.G_AB = nn.DataParallel(self.G_AB.cuda())
        # self.G_BA = nn.DataParallel(self.G_BA.cuda())
        # self.D_A = nn.DataParallel(self.D_A.cuda())
        # self.D_B = nn.DataParallel(self.D_B.cuda())

        # Optimizers

        self.optimizer_G = torch.optim.Adam([{'params': itertools.chain(self.G_AB.parameters(), \
                                                                        self.G_BA.parameters()), \
                                              'initial_lr': 0.0001}], lr=0.0001)
        self.optimizer_D_A = torch.optim.RMSprop(params=[{'params':self.D_A.parameters(), \
                                                          'initial_lr': 0.0001}], \
                                                   lr=0.0001)#RMSprop
        self.optimizer_D_B = torch.optim.RMSprop(params=[{'params': self.D_B.parameters(), \
                                                          'initial_lr': 0.0001}], \
                                                 lr=0.0001)  # RMSprop
        #self.optimizer_D_B = torch.optim.RMSprop(self.D_B.parameters(), lr=0.0001)

        self.scheduler_G = scheduler(self.optimizer_G, step_size=8000, gamma=0.9, last_epoch=-1)#36000
        self.scheduler_D_A = scheduler(self.optimizer_D_A, step_size=8000, gamma=0.9, last_epoch=-1)
        self.scheduler_D_B = scheduler(self.optimizer_D_B, step_size=8000, gamma=0.9, last_epoch=-1)

        #self.FloatTensor = torch.FloatTensor#torch.cuda.FloatTensor
        #self.LongTensor = torch.LongTensor#torch.cuda(self.cudaid).LongTensor

    def compute_gradient_penalty(self,D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
        alpha = alpha.cuda(self.cudaid)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        validity = D(interpolates)
        fake = Variable(torch.FloatTensor(np.ones(validity.shape)).cuda(self.cudaid), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=validity,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def Train(self, turn=2):
        self.shot = -1
        torch.set_grad_enabled(True)

        for t in range(turn):

            # if self.gclip > 0:
            #     utils.clip_grad_value_(self.net.parameters(), self.gclip)

            for kk, (lowImg, highImg) in enumerate(self.dataLoader):

                # torch.cuda.empty_cache()
                self.shot = self.shot + 1
                # torch.cuda.empty_cache()
                # self.scheduler.step()
                self.scheduler_G.step()
                self.scheduler_D_A.step()
                self.scheduler_D_B.step()
                lrImg = (lowImg - lowMean) / lowStd
                mrImg = (highImg - highMean) / highStd
                lrImg = lrImg.cuda(self.cudaid)
                mrImg = mrImg.cuda(self.cudaid)

                if True:
                    self.optimizer_D_A.zero_grad()
                    self.optimizer_D_B.zero_grad()

                    # Generate a batch of images
                    fake_A = self.G_BA(mrImg).detach()
                    fake_B = self.G_AB(lrImg).detach()

                    # ----------
                    # Domain A
                    # ----------

                    # Compute gradient penalty for improved wasserstein training
                    gp_A = self.compute_gradient_penalty(
                        self.D_A, lrImg.data, fake_A.data)
                    # Adversarial loss
                    self.D_A_lrImg = torch.mean(self.D_A(lrImg))
                    self.D_A_fakeA = torch.mean(self.D_A(fake_A))
                    D_A_loss = -self.D_A_lrImg + self.D_A_fakeA + self.lambda_gp * gp_A

                    # ----------
                    # Domain B
                    # ----------

                    # Compute gradient penalty for improved wasserstein training
                    gp_B = self.compute_gradient_penalty(self.D_B, mrImg.data, fake_B.data)
                    # Adversarial loss
                    self.D_B_mrImg = torch.mean(self.D_B(mrImg))
                    self.D_B_fakeB = torch.mean(self.D_B(fake_B))
                    D_B_loss = -self.D_B_mrImg + self.D_B_fakeB + self.lambda_gp * gp_B

                    # Total loss
                    D_loss = D_A_loss + D_B_loss

                    D_loss.backward()
                    self.optimizer_D_A.step()
                    self.optimizer_D_B.step()


                if kk % 2== 0:#
                    print('D_A(fakeA): %f D_B(fakeB):%f' % (self.D_A_fakeA.item(),
                                                            self.D_B_fakeB.item()))
                    print('D_A(lr): %f D_B(mr):%f' % (self.D_A_lrImg.item(),
                                                            self.D_B_mrImg.item()))

                    # ------------------
                    #  Train Generators
                    # ------------------

                    self.optimizer_G.zero_grad()

                    # Translate images to opposite domain
                    fake_A = self.G_BA(mrImg)
                    fake_B = self.G_AB(lrImg)

                    # Reconstruct images
                    recov_A = self.G_BA(fake_B)
                    recov_B = self.G_AB(fake_A)

                    # Adversarial loss
                    G_adv = -torch.mean(self.D_A(fake_A)) - torch.mean(self.D_B(fake_B))
                    # Add super resolution loss
                    #G_sr = self.cycle_loss(fake_B, mrImg) + self.cycle_loss(fake_A, lrImg)
                    # Cycle loss
                    G_cycle = self.cycle_loss(recov_A, lrImg) + self.cycle_loss(recov_B, mrImg)
                    # Total loss
                    #G_loss = self.lambda_adv * G_adv + self.lambda_cycle * G_cycle# + self.lambda_cycle * G_sr
                    G_loss = G_adv + 10.0 * G_cycle  # + self.lambda_cycle * G_sr

                    G_loss.backward()
                    self.optimizer_G.step()

                if self.shot % 5 == 0:

                    lr = self.scheduler_G.get_lr()[0]
                    lossVal = np.float(G_loss.cpu().data.numpy())
                    #print('epoch: %d batch: %d lr:%f loss:%f' % (t, self.shot, lr, lossVal))
                    #print('epoch: %d batch: %d lr:%f loss:%f' % (t, self.shot, self.lr, lossVal))
                    print("\r[Epoch %d] [Batch %d] [LR:%f] [D loss: %f] [G loss: %f, adv: %f,sr: None, cycle: %f]"
                                            % (
                                                t,
                                                self.shot,
                                                lr,
                                                D_loss.item(),
                                                G_loss.item(),
                                                G_adv.item(),
                                                #G_sr.item(),
                                                G_cycle.item()
                                            )
                                        )

                    srImgImg = np.max(lowImg.data.numpy()[0, 0, :, :, :], axis=0)
                    srImgImg = RestoreNetImg(srImgImg,0,1)
                    logger.img('srImg', srImgImg)
                    recovImg = recov_B.cpu().data.numpy()[0, 0, :, :, :]
                    recovImg = RestoreNetImg(recovImg, highMean, highStd)
                    recovImg2XY = np.max(recovImg, axis=0)
                    recovImg2XZ = np.max(recovImg, axis=1)
                    logger.img('recovXY', recovImg2XY)
                    logger.img('recovXZ', recovImg2XZ)
                    reImg = fake_B.cpu().data.numpy()[0, 0, :, :, :]
                    reImg = RestoreNetImg(reImg, highMean, highStd)
                    reImg2XY = np.max(reImg, axis=0)
                    reImg2XZ = np.max(reImg, axis=1)
                    logger.img('reImg2XY', reImg2XY)
                    logger.img('reImg2XZ', reImg2XZ)
                    # interpolate
                    lrImg2 = lowImg.cpu().data.numpy()[0, 0, :, :, :]
                    zoom2 = RestoreNetImg(lrImg2, 0, 1)
                    zoom2 = np.minimum(zoom(zoom2, (3,3,3)), 255)

                    zoom2XY = np.max(zoom2, axis=0)
                    logger.img('zoom2XY', zoom2XY)
                    zoom2XZ = np.max(zoom2, axis=1)
                    logger.img('zoom2XZ', zoom2XZ)
                    highImgXY = np.max(highImg.data.numpy()[0, 0, :, :, :], axis=0)
                    highImgXY = RestoreNetImg(highImgXY, highMean, highStd)
                    logger.img('highImgXY', highImgXY)
                    highImgXZ = np.max(highImg.cpu().data.numpy()[0, 0, :, :, :], axis=1)
                    highImgXZ = RestoreNetImg(highImgXZ, highMean, highStd)
                    logger.img('highImgXZ', highImgXZ)
                    lossVal = np.float(G_loss.cpu().data.numpy())
                    if lossVal > 10000:
                        print('G loss > 10000')
                    else:
                        logger.plot('G_loss', lossVal)

                    lossVal = np.float(D_loss.cpu().data.numpy())
                    if lossVal > 10000:
                        print('D loss > 10000')
                    else:
                        logger.plot('D_loss', lossVal)
                if self.shot != 0 and self.shot % 1000 == 0:
                    if not os.path.exists('saved_models/'):
                        os.mkdir('saved_models/')
                    torch.save(self.G_AB.state_dict(), "saved_models/G_AB_%d.pth" % ( self.shot))
                    torch.save(self.G_BA.state_dict(), "saved_models/G_BA_%d.pth" % ( self.shot))
                    torch.save(self.D_A.state_dict(), "saved_models/D_A_%d.pth" % ( self.shot))
                    torch.save(self.D_B.state_dict(), "saved_models/D_B_%d.pth" % ( self.shot))
        #torch.save(self.net, 'DualGAN.pt')





