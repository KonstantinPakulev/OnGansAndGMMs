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



# Batch size during training
BATCH_SIZE = 128

#Size of preprocessed image
PICTURE_SIZE = 64

# Size of z latent vector
Z_LATENT = 100

# Size of generator output (channels)
GENERATOR_PARAMETER = 64

#  Size of discriminator output (channels)
DISCRIMINATOR_PARAMETER = 64

# Number of epochs to train
NUM_EPOCHS = 25

# Learning rate for optimizers
lr = 0.0002

# Path for logs
LOG_PATH = './dcgan_train_process_1'
os.makedirs(LOG_PATH,exist_ok=True)

# Path for dataset
DATA_PATH = './data'



# Log files
log_discriminator = open(os.path.join(LOG_PATH,'log_discriminator.txt'),'w')
log_discriminator.close()
log_generator = open(os.path.join(LOG_PATH,'log_generator.txt'),'w')
log_generator.close()



transform = transforms.Compose(
    [transforms.CenterCrop(178),
    transforms.Resize(PICTURE_SIZE),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CelebA(root=DATA_PATH, split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

testset = torchvision.datasets.CelebA(root=DATA_PATH, split='test',
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class Generator(nn.Module):
    def __init__(self, architecture=GENERATOR_PARAMETER, hid_dim=Z_LATENT):
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
    def __init__(self, architecture=DISCRIMINATOR_PARAMETER):
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
    def __init__(self, gen_architecture=GENERATOR_PARAMETER, disc_architecture=DISCRIMINATOR_PARAMETER, hid_dim=Z_LATENT,device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(architecture=gen_architecture, hid_dim=hid_dim).to(device)
        self.D = Discriminator(architecture=disc_architecture).to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
model = DCGAN(GENERATOR_PARAMETER, DISCRIMINATOR_PARAMETER,Z_LATENT,device)

model.G.apply(weights_init)
model.D.apply(weights_init)

criterion = nn.BCELoss()

# Evaluation generator data
generator_eval = torch.randn(64, Z_LATENT, 1, 1, device=device)

real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
d_optimizer = optim.Adam(model.D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = optim.Adam(model.G.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop

# Lists to keep track of progress
iters = 0

print("Train has started")

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        data,_= data
        model.D.zero_grad()
        # Format batch
        real_data = data.to(device)

        label = torch.full((real_data.shape[0],), real_label, device=device)
        output = model.D(real_data).view(-1)
        d_loss_real = criterion(output, label)
        d_loss_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(real_data.shape[0], Z_LATENT, 1, 1, device=device)
        gen_data = model.G(noise)
        label.fill_(fake_label)
        output = model.D(gen_data.detach()).view(-1)
        d_loss_gen = criterion(output, label)
        d_loss_gen.backward()
        D_G_z1 = output.mean().item()
        d_loss = d_loss_real + d_loss_gen
        d_optimizer.step()


        model.G.zero_grad()
        label.fill_(real_label)
        output = model.D(gen_data).view(-1)
        g_loss = criterion(output, label)
        g_loss.backward()
        D_G_z2 = output.mean().item()
        g_optimizer.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.8f / %.8f'
                  % (epoch, NUM_EPOCHS, i, len(trainloader),
                     d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
            data = np.transpose(model.G(generator_eval)[0].detach().cpu().numpy(),(1,2,0))
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            image = (norm(data))
            plt.imsave(os.path.join(LOG_PATH,'current_res.png'), image)
        with open(os.path.join(LOG_PATH,'log_discriminator.txt'),'a') as f:
            f.write(str(epoch)+'\t'+str(i)+'\t'+str(d_loss.item())+'\n')
        with open(os.path.join(LOG_PATH,'log_generator.txt'),'a') as f:
            f.write(str(epoch)+'\t'+str(i)+'\t'+str(g_loss.item())+'\n')


        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 10 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(trainloader)-1)):
            with torch.no_grad():
                fake = model.G(generator_eval).detach().cpu()
            image = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(),(1,2,0))
            plt.imsave(os.path.join(LOG_PATH,str(iters)+'.png'), image)
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(trainloader)-1)):
                torch.save(model.G.state_dict(), os.path.join(LOG_PATH,str(iters)+'_generator.ckpt'))
                torch.save(model.D.state_dict(), os.path.join(LOG_PATH,str(iters)+'_discriminator.ckpt'))

        iters += 1