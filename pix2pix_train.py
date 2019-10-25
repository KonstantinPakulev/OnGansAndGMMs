import os
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import upsample
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms, ToPILImage, ToTensor, RandomCrop, Resize

from source.mfa.celeba_dataset import CelebaDataset, FlattenTransform, TRAIN
from source.mfa.mfa_utils import RUN_DIR
from source.mfa.mfa_torch import init_raw_parms_from_gmm, generate_from_posterior
from source.mfa.mfa import MFA

from source.pix2pix.pix2pix import Pix2PixDisc, Pix2PixGen

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--mfa_model_path", type=str)

    args = parser.parse_args()

    DATASET_ROOT = args.dataset_root
    MFA_MODEL_PATH = args.mfa_model_path
    BATCH_SIZE = 10
    NUM_EPOCHS = 10

    train_transforms = transforms.Compose([
        ToPILImage(),
        Resize((64, 64)),
        ToTensor(),
        FlattenTransform()
    ])

    train_dataset = CelebaDataset(DATASET_ROOT, TRAIN, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    fa_gmm_model = MFA()
    fa_gmm_model.load(MFA_MODEL_PATH)

    G_PI, G_MU, G_A, G_D = init_raw_parms_from_gmm(fa_gmm_model)

    G = Pix2PixGen().cuda()
    D = Pix2PixDisc().cuda()

    G.train()
    D.train()

    # loss
    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()

    # Adam optimizer
    G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(NUM_EPOCHS):
        disc_loss = 0
        gen_loss = 0
        count = 0

        for y in train_loader:
            D.zero_grad()

            y = y.cuda()
            x = generate_from_posterior(y, G_PI, G_MU, G_A, G_D).detach()

            y = upsample(y.view(-1, 3, 64, 64), size=256)
            x = upsample(x.view(-1, 3, 64, 64), size=256)

            # x - mfa image
            # y - original image

            D_result = D(x, y).squeeze()
            D_real_loss = BCE_loss(D_result, torch.ones(D_result.size()).cuda())

            G_result = G(x)
            D_result = D(x, G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, torch.zeros(D_result.size()).cuda())

            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()

            G.zero_grad()

            G_result = G(x)
            D_result = D(x, G_result).squeeze()

            G_train_loss = BCE_loss(D_result, torch.ones(D_result.size()).cuda()) + 100 * L1_loss(G_result, y)
            G_train_loss.backward()
            G_optimizer.step()

            disc_loss += D_train_loss
            gen_loss += G_train_loss
            count += 1

            if count % 1000 == 0:
                # Save image
                saved_org_img_path = os.path.join(RUN_DIR, f"orig_image_1.png")
                saved_gen_img_path = os.path.join(RUN_DIR, f"gen_image_1.png")

                plt.imsave(saved_org_img_path, x[0].cpu().numpy().transpose(2, 1, 0).clip(0, 1))
                plt.imsave(saved_gen_img_path, G_result.detach()[0].cpu().numpy().transpose(2, 1, 0).clip(0, 1))

                # Save model
                saved_gen_path = os.path.join(RUN_DIR, f"e{epoch + 1}_pix2pix_gen_1")
                torch.save(G.state_dict(), saved_gen_path)

        print(f"Epoch {epoch + 1}; Disc loss: {disc_loss / count}; Val loss: {gen_loss / count}")
