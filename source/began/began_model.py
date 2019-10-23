import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, ndf=4, hid_dim=100):
        super(Encoder, self).__init__()
        self.ndf = ndf
        self.hid_dim = hid_dim
        # YOUR FAVORITE nn.Sequatial here
        self.encoder = nn.Sequential(  # input is (4) x 64 x 64
            nn.Conv2d(3, ndf, 3, 1, 1, bias=False),  # (w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),  # 32-4+2)
            nn.ELU(),
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.ELU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),  # (w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ndf * 2, ndf * 3, 3, 1, 1, bias=False),  # 32-4+2)
            nn.ELU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(ndf * 3, ndf * 3, 3, 1, 1, bias=False),  # (w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ndf * 3, ndf * 3, 3, 1, 1, bias=False),  # 32-4+2)
            nn.ELU())
        self.fc_enc = nn.Linear(16 * 16 * 3 * ndf, hid_dim)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        enc_conv = self.encoder(x)
        z = F.elu(self.fc_enc(enc_conv.view(-1, 16 * 16 * 3 * self.ndf)))
        return z


class Decoder(nn.Module):
    def __init__(self, ndf=4, hid_dim=100):
        super(Decoder, self).__init__()
        self.ndf = ndf
        self.hid_dim = hid_dim
        # YOUR FAVORITE nn.Sequatial here
        ngf = 3 * ndf
        self.ngf = ngf
        self.decoder = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),  # (w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),  # 32-4+2)
            nn.ELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(ngf, 2 * ngf, 1, 1, 0, bias=False),

            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),  # (w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),  # 32-4+2)
            nn.ELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(ngf, 2 * ngf, 1, 1, 0, bias=False),

            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),  # (w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),  # 32-4+2)
            nn.ELU(),

            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.fc_dec = nn.Linear(hid_dim, 16 * 16 * 3 * ndf)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, z):
        dec = (F.relu(self.fc_dec(z))).view(-1, 3 * self.ndf, 16, 16)
        dec_conv = self.decoder(dec)
        return dec_conv


class GAN(nn.Module):
    def __init__(self, ndf=4, hid_dim=100):
        super(GAN, self).__init__()
        self.ndf = ndf
        self.hid_dim = hid_dim
        # YOUR FAVORITE nn.Sequatial here
        self.encoder = Encoder(ndf=ndf, hid_dim=hid_dim)
        ngf = 3 * ndf
        self.ngf = ngf
        self.decoder = Decoder(ndf=ndf, hid_dim=hid_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))
