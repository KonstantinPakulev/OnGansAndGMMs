from argparse import ArgumentParser
import torch
from torch.autograd import Variable

from source.wgangp.celeba_utils import Celebaloader
from source.wgangp.wgan_model import WGAN_GP

from tqdm import tqdm_notebook
from IPython.display import clear_output
from skimage import img_as_ubyte
import imageio
import matplotlib.pyplot as plt


def parse():
  parser = ArgumentParser()

  parser.add_argument("-e", "--n_epochs", dest="N_EPOCHS",
                      help="number of train epochs",default = 100,type=int)

  parser.add_argument("-ld", "--latent_dim",
                      dest="LATENT_DIM", default=100,
                      help="dimension of latent space",type=int)
  
  parser.add_argument("-b", "--batch_size",
                      dest="BATCH_SIZE", default=128,
                      help="size of train and test batches",type=int)

  parser.add_argument("-l", "--learn_rate",
                      dest="LEARN_RATE", default=2e-4,
                      help="leraning rate for optimizer",type=float)

  parser.add_argument("-i", "--start_epoch",
                      dest="STARTING_EPOCH", default=0,
                      help="starting training iteration",type=int)
  
  parser.add_argument("-nc", "--n_critic",
                      dest="N_CRITIC", default=5,
                      help="number of iteration of critic pre 1 generator update",type=int)
  
  parser.add_argument("-s", "--epoch_start",
                      dest="STARTING_EPOCH", default=0,
                      help="starting training epoch",type=int)

  parser.add_argument("-c", "--continue_train",
                      dest="RETRAIN", default=False,
                      help="train continue flag",type=bool)

  parser.add_argument("-d", "--data_folder",
                      dest="DATA_FOLDER", default='./',
                      help="Folder where all results will be located",type=str)

  parser.add_argument("-v", "--verb_iter",
                      dest="VERB_ITER", default=30,
                      help="number of iteration util showing training stage",type=int)
  
  parser.add_argument("-ci", "--check_iter",
                      dest="CHECK_ITER", default=500,
                      help="number of iteration util saving training stage",type=int)
  
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse()
  
  trainset, trainloader, testset, testloader = Celebaloader(DATA_PATH=args.DATA_FOLDER, 
                                                            BATCH_SIZE=args.BATCH_SIZE)
  
  epochs = args.N_EPOCHS
  lr = args.LEARN_RATE
  n_critic = args.N_CRITIC
  z_dim = args.LATENT_DIM
  
  device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

  model = WGAN_GP(3, z_dim)
  model.to(device)
  
  if args.RETRAIN:
    checkpoint = torch.load(args.DATA_FOLDER + '/wgangp_gen_{}.pt'.format(args.STARTING_EPOCH - 1))
    model.G.load_state_dict(checkpoint)
    checkpoint = torch.load(args.DATA_FOLDER + '/wgangp_disc_{}.pt'.format(args.STARTING_EPOCH - 1))
    model.D.load_state_dict(checkpoint)

  d_optimizer = torch.optim.Adam(model.D.parameters(), lr=lr, betas=(0.5, 0.999))
  g_optimizer = torch.optim.Adam(model.G.parameters(), lr=lr, betas=(0.5, 0.999))
  
  z_sample = Variable(torch.randn(64, z_dim))
  z_sample = z_sample.to(device)

  D_loss = []
  G_loss = []
  GP = []
  images = []
  lam = 10.

  try:
    for epoch in range(args.STARTING_EPOCH, args.STARTING_EPOCH + epochs):
        for i, (imgs, _) in enumerate(tqdm_notebook(trainloader)):
          step = epoch * len(trainloader) + i + 1

          # set train
          model.G.train()

          # leafs
          imgs = Variable(imgs)
          bs = imgs.size(0)
          z = Variable(torch.randn(bs, z_dim))
          imgs, z = imgs.to(device), z.to(device)

          f_imgs = model.G(z)

          r_logit = model.D(imgs)
          f_logit = model.D(f_imgs.detach())

          wd = r_logit.mean() - f_logit.mean()
          gp = model.gradient_penalty(imgs.data, f_imgs.data)
          d_loss = -wd + gp * lam

          model.D.zero_grad()
          d_loss.backward()
          d_optimizer.step()

          D_loss.append(wd.data.cpu().numpy())
          GP.append(gp.data.cpu().numpy())

          if step % n_critic == 0:
              z = Variable(torch.randn(bs, z_dim)).to(device)
              f_imgs = model.G(z)
              f_logit = model.D(f_imgs)
              g_loss = -f_logit.mean()

              model.D.zero_grad()
              model.G.zero_grad()
              g_loss.backward()
              g_optimizer.step()

              G_loss.append(g_loss.data.cpu().numpy())

          if (i + 1) % args.VERB_ITER // 2 == 0:
              print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, len(trainloader)))

          if (i + 1) % args.VERB_ITER == 0:
              clear_output()
              model.G.eval()

              plt.figure(figsize=(15, 5))
              plt.subplot(1, 2, 1)
              plt.plot(D_loss, label='D loss')
              plt.legend(loc='best')
              plt.subplot(1, 2, 2)
              plt.plot(G_loss, label='G loss')
              plt.legend(loc='best')
              plt.show()
              
              grid = model.sample(z_sample)
              torch.save(model.G.state_dict(), args.DATA_FOLDER + '/wgangp' + '_gen_{}.pt'.format(epoch))
              torch.save(model.D.state_dict(), args.DATA_FOLDER + '/wgangp' + '_disc_{}.pt'.format(epoch))
              
              with open(args.DATA_FOLDER +'/log_discriminator.txt','a') as f:
                  f.write(str(epoch) + '\t' + str(i) + '\t' + str(D_loss[-1])+'\n')
              with open(args.DATA_FOLDER + '/log_generator.txt','a') as f:
                  f.write(str(epoch) + '\t' + str(i) + '\t' + str(G_loss[-1])+'\n')
          if (i + 1) % args.CHECK_ITER == 0:
              images.append(img_as_ubyte(grid))
              imageio.mimsave(args.DATA_FOLDER + '/wgangp_{}.gif'.format(args.STARTING_EPOCH), images)


  except KeyboardInterrupt:
    print('EarlyStop')
    grid = model.sample(z_sample)
    images.append(img_as_ubyte(grid))
    #drive.mount('/content/drive')
    imageio.mimsave(args.DATA_FOLDER + '/wgangp_{}.gif'.format(args.STARTING_EPOCH), images)
    torch.save(model.G.state_dict(), args.DATA_FOLDER + '/wgangp' + '_gen_{}.pt'.format(epoch))
    torch.save(model.D.state_dict(), args.DATA_FOLDER + '/wgangp' + '_disc_{}.pt'.format(epoch))
