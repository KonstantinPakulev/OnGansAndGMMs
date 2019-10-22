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
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", dest="NUM_EPOCHS",
                    help="number of train epochs",default = 500,type=int)
parser.add_argument("-p", "--patience",
                    dest="PATIENCE", default=100,
                    help="number of epochs to wait before early stopping",type=int)
parser.add_argument("-b", "--batch_size",
                    dest="BATCH_SIZE", default=64,
                    help="size of train and test batches",type=int)
parser.add_argument("-t", "--train_num",
                    dest="TRAIN_NUM", default=0,
                    help="number of training current model",type=int)
parser.add_argument("-l", "--learn_rate",
                    dest="LEARN_RATE", default=1e-4,
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
parser.add_argument("-k", "--k",
                    dest="k", default=0,
                    help="k train parameter",type=float)

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
GENERATOR_PARAMETER = 32

# Number of epochs to train
NUM_EPOCHS = args.NUM_EPOCHS

# Started epoch
STARTED_EPOCH = args.STARTED_EPOCH

# Started iteration
STARTED_ITER = args.STARTED_ITER

# k train parameter
k = args.k

# Learning rate for optimizers
lr = args.LEARN_RATE

# Path for logs
LOG_PATH = './began_train_process_1'
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

class Encoder(nn.Module):
    def __init__(self,ndf=4,hid_dim = 100):
        super(Encoder, self).__init__()
        self.ndf = ndf
        self.hid_dim = hid_dim
       # YOUR FAVORITE nn.Sequatial here
        self.encoder = nn.Sequential(# input is (4) x 64 x 64
            nn.Conv2d(3, ndf, 3, 1, 1, bias=False), #(w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),#32-4+2) 
            nn.ELU(),
            nn.Conv2d(ndf , ndf * 2, 3, 1, 1, bias=False),
            nn.ELU(),
            
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False), #(w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ndf * 2, ndf * 3, 3, 1, 1, bias=False),#32-4+2) 
            nn.ELU(),
            
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(ndf * 3, ndf * 3, 3, 1, 1, bias=False), #(w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ndf * 3, ndf * 3, 3, 1, 1, bias=False),#32-4+2) 
            nn.ELU())
        self.fc_enc = nn.Linear(16*16*3*ndf, hid_dim)

        self.init_params()

        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        enc_conv = self.encoder(x)
        z = F.elu(self.fc_enc(enc_conv.view(-1,16*16*3*self.ndf)))
        return z
    
class Decoder(nn.Module):
    def __init__(self,ndf=4,hid_dim = 100):
        super(Decoder, self).__init__()
        self.ndf = ndf
        self.hid_dim = hid_dim
       # YOUR FAVORITE nn.Sequatial here
        ngf = 3*ndf
        self.ngf  = ngf
        self.decoder = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False), #(w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),#32-4+2) 
            nn.ELU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(ngf,2*ngf,1,1,0, bias=False),
            
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False), #(w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),#32-4+2) 
            nn.ELU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(ngf,2*ngf,1,1,0, bias=False),
            
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False), #(w-f+2p)/S+1  (64-4+2)/2+1 = 32*
            nn.ELU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),#32-4+2) 
            nn.ELU(),
            
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.fc_dec = nn.Linear(hid_dim, 16*16*3*ndf)

        self.init_params()

        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                
    def forward(self, z):
        dec =(F.relu(self.fc_dec(z))).view(-1,3*self.ndf,16,16)
        dec_conv = self.decoder(dec)
        return dec_conv
    

class GAN(nn.Module):
    def __init__(self,ndf=4,hid_dim = 100):
        super(GAN, self).__init__()
        self.ndf = ndf
        self.hid_dim = hid_dim
       # YOUR FAVORITE nn.Sequatial here
        self.encoder = Encoder(ndf=ndf,hid_dim=hid_dim)
        ngf = 3*ndf
        self.ngf  = ngf
        self.decoder = Decoder(ndf=ndf,hid_dim=hid_dim)
        

    def forward(self, x):
        return self.decoder(self.encoder(x))
        
Discriminator=GAN(ndf = GENERATOR_PARAMETER, hid_dim = Z_LATENT)
Discriminator.to(device)

Generator = Decoder(ndf = GENERATOR_PARAMETER, hid_dim = Z_LATENT)
Generator.to(device)



if RETRAIN:
    checkpoint = torch.load(os.path.join(LOG_PATH,str(STARTED_ITER-1)+'_generator.ckpt'))
    Generator.load_state_dict(checkpoint)
    checkpoint = torch.load(os.path.join(LOG_PATH,str(STARTED_ITER-1)+'_discriminator.ckpt'))
    Discriminator.load_state_dict(checkpoint)

criterion = nn.L1Loss()
    
# Evaluation generator data
generator_eval = torch.rand(64, Z_LATENT, device=device)


# Setup Adam optimizers for both G and D
d_optimizer = optim.Adam(Discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = optim.Adam(Generator.parameters(), lr=lr, betas=(0.5, 0.999))

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

    def __call__(self, val_loss, Generator, Discriminator):
        
        global MODEL_PATH

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, Generator, Discriminator)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, Generator, Discriminator)
            self.counter = 0

    def save_checkpoint(self, val_loss, Generator, Discriminator):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(Generator.state_dict(), os.path.join(self.model_path,'best_generator.ckpt'))
        torch.save(Discriminator.state_dict(), os.path.join(self.model_path,'best_discriminator.ckpt'))
        self.val_loss_min = val_loss

print("Train has started")

early_stopping = EarlyStopping(patience=PATIENCE, verbose=True,model_path=LOG_PATH)

for epoch in range(STARTED_EPOCH,NUM_EPOCHS+STARTED_EPOCH):
    g_loss_epoch=[]
    for i, data in enumerate(trainloader, 0):
        data,_= data
        Discriminator.zero_grad()
        # Format batch
        real_data = data.to(device)

        output = Discriminator(real_data)
        d_loss_real = criterion(output, real_data)
       
        noise = torch.rand(real_data.shape[0], Z_LATENT, device=device)
        gen_data = Generator(noise)

        output = Discriminator(gen_data.detach())
        d_loss_gen = criterion(output,gen_data)

#         for p in Discriminator.parameters():
#                 if len(p.shape) != 2:
#                     p.data.clamp_(-0.1, 0.1)
        d_loss = d_loss_real - d_loss_gen*k
        d_loss.backward()
        d_optimizer.step()
        

        Generator.zero_grad()
        gen_data = Generator(noise)
        output = Discriminator(gen_data.detach())
        g_loss = criterion(output,gen_data)
        g_loss.backward()

        g_optimizer.step()
        g_loss_epoch.append(g_loss.item())
        
        balance = (0.5*d_loss_real.item() - d_loss_gen.item())
        k+=0.001*balance
        # Output training stats
        if i % 50 == 0:
#             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.8f / %.8f'
#                   % (epoch, NUM_EPOCHS, i, len(trainloader),
#                      d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
            data = np.transpose(Generator(generator_eval)[0].detach().cpu().numpy(),(1,2,0))
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
        writer.add_scalar('/k', k, iters)
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == NUM_EPOCHS+STARTED_EPOCH-1) and (i == len(trainloader)-1)):
            z = []
            for inter in range(10):
                z0 = np.random.uniform(-1,1,100)
                z10 = np.random.uniform(-1,1,100)
                def slerp(val, low, high):
                    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
                    so = np.sin(omega)
                    if so == 0:
                        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
                    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high 

                z.append(z0)
                for i in range(1, 9):
                    z.append(slerp(i*0.1, z0, z10))
                z.append(z10.reshape(1, 100)) 
            z = [_.reshape(1, 100) for _ in z]
            z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float()).to(device)
            with torch.no_grad():
                fake = Generator(z_var).detach().cpu()
            image = np.transpose(vutils.make_grid(fake, padding=2, normalize=True,nrow=10).numpy(),(1,2,0))
            plt.imsave(os.path.join(LOG_PATH,str(iters)+'.png'), image)
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS+STARTED_EPOCH-1) and (i == len(trainloader)-1)):
                torch.save(Generator.state_dict(), os.path.join(LOG_PATH,str(iters)+'_generator.ckpt'))
                torch.save(Discriminator.state_dict(), os.path.join(LOG_PATH,str(iters)+'_discriminator.ckpt'))

        iters += 1
    early_stopping(np.mean(g_loss_epoch), Generator, Discriminator)
    if early_stopping.early_stop:
        print(epoch,"Early stopping")
        break