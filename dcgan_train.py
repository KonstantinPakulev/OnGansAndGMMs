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
from tensorboardX import SummaryWriter

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", dest="NUM_EPOCHS",
                    help="number of train epochs",default = 500,type=int)
parser.add_argument("-p", "--patience",
                    dest="PATIENCE", default=100,
                    help="number of epochs to wait before early stopping",type=int)
parser.add_argument("-b", "--batch_size",
                    dest="BATCH_SIZE", default=128,
                    help="size of train and test batches",type=int)
parser.add_argument("-t", "--train_num",
                    dest="TRAIN_NUM", default=0,
                    help="number of training current model",type=int)
parser.add_argument("-l", "--learn_rate",
                    dest="LEARN_RATE", default=2e-4,
                    help="leraning rate for optimizer",type=float)
parser.add_argument("-i", "--iter_start",
                    dest="STARTED_ITER", default=0,
                    help="started training iteration",type=int)
parser.add_argument("-s", "--epoch_start",
                    dest="STARTED_EPOCH", default=0,
                    help="started training epoch",type=int)
parser.add_argument("-c", "--continue_train",
                    dest="RETRAIN", default=False,
                    help="train continue flag",type=bool)

args = parser.parse_args()

# Training number
TRAIN_NUM=args.TRAIN_NUM

# Batch size during training
BATCH_SIZE = args.BATCH_SIZE

#Size of preprocessed image
PICTURE_SIZE = 64

# Size of z latent vector
Z_LATENT = 100

# Size of generator output (channels)
GENERATOR_PARAMETER = 64

#  Size of discriminator output (channels)
DISCRIMINATOR_PARAMETER = 64

# Number of epochs to train
NUM_EPOCHS = args.NUM_EPOCHS

# Started epoch
STARTED_EPOCH = args.STARTED_EPOCH

# Started iteration
STARTED_ITER = args.STARTED_ITER

# Learning rate for optimizers
lr = 0.0002

# Path for logs
LOG_PATH = './dcgan_train_process_2'
os.makedirs(LOG_PATH,exist_ok=True)

# Path for dataset
DATA_PATH = './data'

# Retrain flag
RETRAIN=args.RETRAIN

# Early Stopping patience
PATIENCE = args.PATIENCE



# Log files
log_discriminator = open(os.path.join(LOG_PATH,str(TRAIN_NUM)+'_log_discriminator.txt'),'w')
log_discriminator.close()
log_generator = open(os.path.join(LOG_PATH,str(TRAIN_NUM)+'_log_generator.txt'),'w')
log_generator.close()

writer=SummaryWriter(LOG_PATH)

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

if RETRAIN:
    checkpoint = torch.load(os.path.join(LOG_PATH,str(STARTED_ITER-1)+'_generator.ckpt'))
    model.G.load_state_dict(checkpoint)
    checkpoint = torch.load(os.path.join(LOG_PATH,str(STARTED_ITER-1)+'_discriminator.ckpt'))
    model.D.load_state_dict(checkpoint)
else:
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

iters = STARTED_ITER

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False,model_path='./'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_path=model_path

    def __call__(self, val_loss, model):
        
        global MODEL_PATH

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.G.state_dict(), os.path.join(self.model_path,'best_generator.ckpt'))
        torch.save(model.D.state_dict(), os.path.join(self.model_path,'best_discriminator.ckpt'))
        self.val_loss_min = val_loss

print("Train has started")

early_stopping = EarlyStopping(patience=PATIENCE, verbose=True,model_path=LOG_PATH)

for epoch in range(STARTED_EPOCH,NUM_EPOCHS+STARTED_EPOCH):
    g_loss_epoch=[]
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
        for p in model.D.parameters():
                if len(p.shape) != 2:
                    p.data.clamp_(-0.1, 0.1)
        d_loss = d_loss_real + d_loss_gen
        d_optimizer.step()
        

        model.G.zero_grad()
        label.fill_(real_label)
        output = model.D(gen_data).view(-1)
        g_loss = criterion(output, label)
        g_loss.backward()
        D_G_z2 = output.mean().item()
        g_optimizer.step()
        g_loss_epoch.append(g_loss.item())
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.8f / %.8f'
                  % (epoch, NUM_EPOCHS, i, len(trainloader),
                     d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
            data = np.transpose(model.G(generator_eval)[0].detach().cpu().numpy(),(1,2,0))
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            image = (norm(data))
            plt.imsave(os.path.join(LOG_PATH,'current_res.png'), image)
        with open(os.path.join(LOG_PATH,str(TRAIN_NUM)+'_log_discriminator.txt'),'a') as f:
            f.write(str(epoch)+'\t'+str(i)+'\t'+str(d_loss.item())+'\n')
        with open(os.path.join(LOG_PATH,str(TRAIN_NUM)+'_log_generator.txt'),'a') as f:
            f.write(str(epoch)+'\t'+str(i)+'\t'+str(g_loss.item())+'\n')

        writer.add_scalar('/Loss/Generator', g_loss.item(), iters)
        writer.add_scalar('/Loss/Discriminator', d_loss.item(), iters)
        writer.add_scalar('/Loss/Discriminator_real', d_loss_real.item(), iters)
        writer.add_scalar('/Loss/Discriminator_fake', d_loss_gen.item(), iters)
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == NUM_EPOCHS+STARTED_EPOCH-1) and (i == len(trainloader)-1)):
            with torch.no_grad():
                fake = model.G(generator_eval).detach().cpu()
            image = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(),(1,2,0))
            plt.imsave(os.path.join(LOG_PATH,str(iters)+'.png'), image)
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS+STARTED_EPOCH-1) and (i == len(trainloader)-1)):
                torch.save(model.G.state_dict(), os.path.join(LOG_PATH,str(iters)+'_generator.ckpt'))
                torch.save(model.D.state_dict(), os.path.join(LOG_PATH,str(iters)+'_discriminator.ckpt'))

        iters += 1
    early_stopping(np.mean(g_loss_epoch), model)
    if early_stopping.early_stop:
        print(epoch,"Early stopping")
        break