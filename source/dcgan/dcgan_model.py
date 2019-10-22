import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn as nn
import torch.optim as optim
import os

class Generator(nn.Module):
    def __init__(self, architecture=3, hid_dim=100):
        super().__init__()

        self.gen = nn.Sequential(
            nn.ConvTranspose2d( hid_dim, architecture * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(architecture * 8),
            nn.ReLU(True),#(architecture*8)*4*4
            nn.ConvTranspose2d(architecture * 8, architecture * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(architecture * 4),
            nn.ReLU(True),#(architecture*4)*4*4
            nn.ConvTranspose2d( architecture * 4, architecture * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(architecture * 2),
            nn.ReLU(True),#(architecture*2)*16*16
            nn.ConvTranspose2d( architecture * 2, architecture, 4, 2, 1, bias=False),
            nn.BatchNorm2d(architecture),
            nn.ReLU(True),#architecture*32*32
            nn.ConvTranspose2d( architecture, 3, 4, 2, 1, bias=False),
            nn.Tanh()#3*64*64
        )

    def forward(self, x):
        generation = self.gen(x)
        return generation


class Discriminator(nn.Module):
    def __init__(self, architecture=3):
        super().__init__()
        self.disc = nn.Sequential(#3*64*64
            nn.Conv2d(3, architecture, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),#architecture*32*32
            nn.Conv2d(architecture, architecture * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(architecture * 2),
            nn.LeakyReLU(0.2, inplace=True),#(architecture*2)*16*16
            nn.Conv2d(architecture * 2, architecture * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(architecture * 4),
            nn.LeakyReLU(0.2, inplace=True),#(architecture*4)*8*8
            nn.Conv2d(architecture * 4, architecture * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(architecture * 8),
            nn.LeakyReLU(0.2, inplace=True),#(architecture*8)*4*4
            nn.Conv2d(architecture * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        output = self.disc(x)
        return output


class DCGAN(nn.Module):
    def __init__(self, gen_architecture=3, disc_architecture=3, hid_dim=100,device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(architecture=gen_architecture, hid_dim=hid_dim).to(device)
        self.D = Discriminator(architecture=disc_architecture).to(device)