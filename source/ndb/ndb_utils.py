import numpy as np

import torch


def norm_ip(img, min, max):
    img = img.clamp(min=min, max=max)
    img = img.add(-min).div(max - min + 1e-5)
    return img


def norm_range(t):
    return norm_ip(t, float(t.min()), float(t.max()))


def sample_from_gan(generator, z_dim, num_test, samples_per_iter=1000):
    samples_list = []

    with torch.no_grad():
        for i in range(num_test // samples_per_iter):
            i_samples = generator(torch.randn(samples_per_iter, z_dim).cuda())
            samples_list.append(norm_range(i_samples).
                                clamp(0, 1).view(samples_per_iter, -1).detach().cpu().numpy())

    return np.concatenate(samples_list, axis=0)


def sample_from_gan_v2(generator, z_dim, num_test, samples_per_iter=1000):
    samples_list = []

    with torch.no_grad():
        for i in range(num_test // samples_per_iter):
            i_samples = generator(torch.randn(samples_per_iter, z_dim, 1, 1).cuda())
            samples_list.append(norm_range(i_samples).
                                clamp(0, 1).view(samples_per_iter, -1).detach().cpu().numpy())

    return np.concatenate(samples_list, axis=0)
