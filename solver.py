import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
#from torch.utils.tensorboard import SummaryWriter
from fastprogress import progress_bar

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models.pix2pix import GeneratorUNet, Discriminator, weights_init_normal
from DataLoder import ImageDataset

import torch.nn as nn
import torch.nn.functional as F
import torch

def load_model(cuda, model_path):
    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    if model_path:
        # Load pretrained models
        generator.load_state_dict(torch.load(os.path.join(model_path, "generator.pth")))
        discriminator.load_state_dict(torch.load(os.path.join(model_path, "discriminator.pth")))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
    return generator, discriminator
    
class Trainer():
    def __init__(self, cfg, num_workers, pin_memory):
        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()
        self.ck = cfg.ck
        self.tb = cfg.tb
        if not os.path.exists(self.ck):
            os.makedirs(self.ck, exist_ok=True)
            os.makedirs(self.tb, exist_ok=True)
 
        # data loader
        dataset = ImageDataset(cfg.train_path, val_path=cfg.test_path, transform_=True, mode="train")
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        
        val_dataset = ImageDataset(cfg.test_path, val_path=None, transform_=True, mode="val")
        self.val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=1, pin_memory=pin_memory)
        self.writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR,
                           filename_suffix=f'OPT_LR_{cfg.lr}_BS_Size_{cfg.img_width}',
                           comment=f'OPT_LR_{cfg.lr}_BS_Size_{cfg.img_width}')
        

    def sample_images(self, generator, itr, epoch):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(self.val_dataloader))
        real_A = Variable(imgs["B"].type(Tensor))
        real_B = Variable(imgs["A"].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, os.path.join(self.ck, '{}_{}'.format(itr, epoch)+"_images.png"), nrow=5, normalize=True)

    def start_training(self, cfg, cuda, model_path=None):
        global Tensor
        # Tensor type
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        generator, discriminator = load_model(cuda, model_path)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        if cuda:
            self.criterion_GAN.cuda()
            self.criterion_pixelwise.cuda()
        # Loss weight of L1 pixel-wise loss between translated image and real image
        lambda_pixel = 100
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, cfg.img_height // 2 ** 4, cfg.img_width // 2 ** 4)

        prev_time = time.time()
        cur = 0
        for epoch in range(cfg.epoch, cfg.n_epochs):
            i = 0
            for batch in progress_bar(self.dataloader):
                # Model inputs
                real_A = Variable(batch["B"].type(Tensor))
                real_B = Variable(batch["A"].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # GAN loss
                fake_B = generator(real_A)
                pred_fake = discriminator(fake_B, real_A)
                loss_GAN = self.criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = self.criterion_pixelwise(fake_B, real_B)

                # Total loss
                loss_G = loss_GAN + lambda_pixel * loss_pixel

                loss_G.backward()

                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(real_B, real_A)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(fake_B.detach(), real_A)
                loss_fake = self.criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                batches_left = cfg.n_epochs * len(self.dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                self.writer.add_scalar("loss_D", loss_D.item(), cur)
                self.writer.add_scalar("loss_G", loss_G.item(), cur)
                self.writer.add_scalar("loss_pixel", loss_pixel.item(), cur)
                self.writer.add_scalar("loss_GAN", loss_GAN.item(), cur)
                
                i += 1
                cur += 1
                # If at sample interval save image
                if cur % cfg.sample_interval == 0:
                    self.sample_images(generator, cur, epoch)

            torch.save(generator.state_dict(), os.path.join(self.tb, "generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(self.tb, "discriminator.pth"))

