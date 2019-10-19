import os
from argparse import ArgumentParser

import torch
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms, ToPILImage, ToTensor, RandomCrop, Resize

from source.celeba_dataset import CelebaDataset, FlattenTransform, TRAIN, VAL
from source.mfa_utils import get_dataset_mean_and_std, gmm_initial_guess, get_random_samples, \
    RUN_DIR, INIT_GMM_FILE, SAVED_GMM_FILE
from source.mfa_torch import get_log_likelihood, init_raw_parms_from_gmm, raw_to_gmm
from source.mfa import MFA


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str)

    args = parser.parse_args()

    NUM_COMPONENTS = 200
    LATENT_DIMENSION = 10
    INIT_METHOD = 'km'

    BATCH_SIZE = 150

    LR = 1e-4
    NUM_EPOCHS = 2

    DATASET_ROOT = args.dataset_root

    train_transforms = transforms.Compose([
        ToPILImage(),
        RandomCrop((78, 103)),
        Resize((64, 64)),
        ToTensor(),
        FlattenTransform()
    ])

    train_dataset = CelebaDataset(DATASET_ROOT, TRAIN, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_transforms = transforms.Compose([
        ToPILImage(),
        Resize((64, 64)),
        ToTensor(),
        FlattenTransform()
    ])

    val_dataset = CelebaDataset(DATASET_ROOT, VAL, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    init_gmm_path = os.path.join(RUN_DIR, INIT_GMM_FILE)
    saved_gmm_path = os.path.join(RUN_DIR, SAVED_GMM_FILE)

    if not os.path.exists(RUN_DIR):
        os.mkdir(RUN_DIR)

    if os.path.exists(init_gmm_path):
        init_gmm = MFA()
        init_gmm.load(init_gmm_path)
    else:
        print('Initial guess...')
        _, init_std = get_dataset_mean_and_std(DATASET_ROOT)

        init_samples_per_comp = 300
        init_n = min(train_dataset.__len__(), max(NUM_COMPONENTS * init_samples_per_comp, 10000))

        print('Collecting an initial sample of', init_n, 'samples...')

        init_samples = get_random_samples(DATASET_ROOT, init_n).detach().numpy()
        init_gmm = gmm_initial_guess(init_samples, NUM_COMPONENTS, LATENT_DIMENSION,
                                     clustering_method=INIT_METHOD,
                                     component_model='fa', dataset_std=init_std, default_noise_std=0.15)

        init_gmm.save(init_gmm_path)

    print("Starting training...")

    G_PI, G_MU, G_A, G_D = init_raw_parms_from_gmm(init_gmm)
    theta_G = [G_PI, G_MU, G_A, G_D]

    optimizer = Adam(theta_G, lr=LR)

    for i in range(NUM_EPOCHS):
        train_loss = 0
        count = 0

        for x in train_loader:
            optimizer.zero_grad()

            loss = -get_log_likelihood(x.cuda(), *theta_G)

            loss.backward()

            optimizer.step()

            train_loss += loss.cpu().detach().item()
            count += 1

        train_loss /= count

        # Calculate validation loss
        val_loss = 0
        count = 0

        with torch.no_grad():
            for x in val_loader:
                loss = -get_log_likelihood(x.cuda(), *theta_G)

                val_loss += loss.cpu().detach().item()
                count += 1

        val_loss /= count

        print(f"Epoch {i + 1}; Train loss: {train_loss}; Val loss: {val_loss}")

        # Save model
        saved_gmm = raw_to_gmm(G_PI.cpu().detach().numpy(),
                               G_MU.cpu().detach().numpy(),
                               G_A.cpu().detach().numpy(),
                               G_D.cpu().detach().numpy())
        saved_gmm.save(saved_gmm_path)

