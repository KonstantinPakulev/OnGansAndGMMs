import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt


def D_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, 2, 2),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2))


class Discriminator(nn.Module):
    def __init__(self, in_channels, hid_channels=64):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 5, 2, 2),
            nn.LeakyReLU(0.2),
            D_conv_block(hid_channels, hid_channels * 2),
            D_conv_block(hid_channels * 2, hid_channels * 4),
            D_conv_block(hid_channels * 4, hid_channels * 8),
            nn.Conv2d(hid_channels * 8, 1, 4))

    def forward(self, x):
        y = self.convs(x)
        y = y.view(-1)
        return y


def G_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 5, 2, padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class Generator(nn.Module):
    def __init__(self, latent_dim, hid_channels=64):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hid_channels * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(hid_channels * 8 * 4 * 4),
            nn.ReLU())
        self.convs = nn.Sequential(
            G_conv_block(hid_channels * 8, hid_channels * 4),
            G_conv_block(hid_channels * 4, hid_channels * 2),
            G_conv_block(hid_channels * 2, hid_channels),
            nn.ConvTranspose2d(hid_channels, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.fc(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.convs(y)
        return y


class WGAN_GP(nn.Module):
    def __init__(self, in_channels, latent_dim, d_channels=64, g_channels=64):
        super(WGAN_GP, self).__init__()

        self.G = Generator(latent_dim, hid_channels=g_channels)
        self.D = Discriminator(in_channels, hid_channels=d_channels)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def gradient_penalty(self, x, y):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        # interpolation
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape).to(device)
        z = x + alpha * (y - x)

        # gradient penalty
        z = Variable(z, requires_grad=True).to(device)
        o = self.D(z)
        g = grad(o, z, grad_outputs=torch.ones(o.size()).to(device), create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

        return gp

    def sample(self, z, show=True):
        grid = make_grid(self.G(z).cpu().data, normalize=True)
        if show:
            plt.figure(figsize=(7, 7))
            grid = np.transpose(grid.numpy().clip(0, 1), (1, 2, 0))
            plt.imshow(grid)
            plt.show()
        return grid
