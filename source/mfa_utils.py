import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms, ToPILImage, ToTensor, Resize

from source.mfa import MFA
from source.celeba_datasets import CelebaDataset, FlattenTransform, TRAIN

RUN_DIR = './run'

TRAINING_MEAN_FILE = 'training_mean.npy'
TRAINING_STD_FILE = 'training_std.npy'
INIT_GMM_FILE = 'init_gmm.pkl'
SAVED_GMM_FILE = 'saved_gmm.pkl'


class Timer(object):
    def __init__(self, name='Operation'):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('%s took: %s sec' % (self.name, time.time() - self.tstart))


def kmeans_clustering(samples, num_clusters, get_centers=False):
    N, d = samples.shape
    K = num_clusters
    # Select random d_used coordinates out of d
    d_used = min(d, max(500, d // 8))
    d_indices = np.random.choice(d, d_used, replace=False)
    print('Performing k-means clustering to {} components of {} samples in dimension {}/{} ...'.format(K, N, d_used, d))
    with Timer('K-means'):
        clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(samples[:, d_indices])
    labels = clusters.labels_
    if get_centers:
        centers = np.zeros([K, d])
        for i in range(K):
            centers[i, :] = np.mean(samples[labels == i, :], axis=0)
        return labels, centers
    return labels


def gmm_initial_guess(samples, num_components, latent_dim, clustering_method='km', component_model='fa',
                      default_noise_std=0.5, dataset_std=1.0):
    N, d = samples.shape
    components = {}
    if clustering_method == 'rnd':
        # In random mode, l+1 samples are randomly selected per component, a plane is fitted through them and the
        # noise variance is set to the default value.
        print('Performing random-selection and FA/PPCA initialization...')
        for i in range(num_components):
            fa = FactorAnalysis(latent_dim)
            used_samples = np.random.choice(N, latent_dim + 1, replace=False)
            fa.fit(samples[used_samples])
            components[i] = {'A': fa.components_.T, 'mu': fa.mean_,
                             'D': np.ones([d]) * np.power(default_noise_std, 2.0),
                             'pi': 1.0 / num_components}
    elif clustering_method == 'km':
        # In k-means mode, the samples are clustered using k-means and a PPCA or FA model is then fitted for each cluster.
        labels = kmeans_clustering(samples / dataset_std, num_components)
        print("Estimating Factor Analyzer parameters for each cluster")
        components = {}
        for i in range(num_components):
            print('.', end='', flush=True)
            if component_model == 'fa':
                model = FactorAnalysis(latent_dim)
                model.fit(samples[labels == i])
                components[i] = {'A': model.components_.T, 'mu': model.mean_, 'D': model.noise_variance_,
                                 'pi': np.count_nonzero(labels == i) / float(N)}
            elif component_model == 'ppca':
                model = PCA(latent_dim)
                model.fit(samples[labels == i])
                components[i] = {'A': model.components_.T, 'mu': model.mean_,
                                 'D': np.ones([d]) * model.noise_variance_ / d,
                                 'pi': np.count_nonzero(labels == i) / float(N)}
            else:
                print('Unknown component model -', component_model)
        print()
    else:
        print('Unknown clustering method -', clustering_method)
    return MFA(components)


def get_random_samples(dataset_root, num_samples):
    transform = transforms.Compose([
        ToPILImage(),
        Resize((64, 64)),
        ToTensor(),
        FlattenTransform()
    ])

    dataset = CelebaDataset(dataset_root, TRAIN, transform)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    return loader.__iter__().__next__()


def get_dataset_mean_and_std(dataset_root, num_samples=20000):
    mean_path = os.path.join(RUN_DIR, TRAINING_MEAN_FILE)
    std_path = os.path.join(RUN_DIR, TRAINING_STD_FILE)

    if not os.path.isfile(mean_path):
        print('Calculating dataset mean and std...')

        samples = get_random_samples(dataset_root, num_samples).detach().numpy()
        dataset_mean = np.mean(samples, axis=0)
        dataset_std = np.std(samples - dataset_mean, axis=0)

        np.save(mean_path, dataset_mean)
        np.save(std_path, dataset_std)
    else:
        dataset_mean = np.load(mean_path)
        dataset_std = np.load(std_path)

    return dataset_mean, dataset_std


def plot_figures(figures, nrows=1, ncols=1, size=None):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axes_list = plt.subplots(ncols=ncols, nrows=nrows, figsize=size)
    for ind, title in zip(range(len(figures)), figures):
        if nrows * ncols != 1:
            axes_list.ravel()[ind].imshow(figures[title], cmap=plt.jet())
            axes_list.ravel()[ind].set_title(title)
            axes_list.ravel()[ind].set_axis_off()
        else:
            axes_list.imshow(figures[title], cmap=plt.jet())
            axes_list.set_title(title)
            axes_list.set_axis_off()

    plt.tight_layout()  # optional


def sample_from_mfa_and_plot(mfa, grid_size=3, image_size=(64, 64), fig_size=(18, 18)):
    num_samples = grid_size ** 2

    samples = mfa.draw_samples(num_samples, False)
    samples = samples.reshape([samples.shape[0], 3, image_size[0], image_size[1]]).\
        transpose([0, 2, 3, 1]).clip(0, 1)

    figures = {}
    for i in range(num_samples):
        figures[i] = samples[i]

    plot_figures(figures, grid_size, grid_size, fig_size)

