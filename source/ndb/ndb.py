import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans

from source.mfa.mfa_utils import from_torch_to_numpy, plot_figures


# images should be flatten and ndarray type.
# Supported only for the whole datasets
class NDB():
    def __init__(self, train_data, k_clusters, alpha=0.05, eps=1e-3, verbose=False):
        self.whitened, self.mean, self.std = self.__whitening(train_data, eps=eps)

        self.k_clusters = k_clusters
        self.t_alpha = norm.ppf(1 - alpha)
        self.verbose = verbose
        self.kmeans = None
        self.train_cnts = None

    def __clustering(self):
        if self.kmeans == None:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k_clusters,
                                          max_iter=500,
                                          batch_size=500,
                                          n_init=10,
                                          verbose=self.verbose,
                                          random_state=0)
        clusters = self.kmeans.fit(self.whitened)
        label_vals, label_counts = np.unique(clusters.labels_, return_counts=True)
        self.bin_order = np.argsort(-label_counts)

        if self.verbose:
            print('Clustered into %d clusters' % self.k_clusters)

        labels_cnt = np.zeros(self.k_clusters)
        labels_cnt[label_vals] = label_counts
        self.train_cnts = labels_cnt
        pass

    def __calc_test_proportion(self, test_data):
        whitened = (test_data - self.mean) / self.std
        labels_cnt = np.zeros(self.k_clusters)
        test_labels = self.kmeans.predict(whitened)
        label_vals, label_counts = np.unique(test_labels, return_counts=True)
        labels_cnt[label_vals] = label_counts

        order = np.argsort(label_counts)

        self.min_labels = label_vals[order][:5]
        self.max_labels = label_vals[order][-5:]

        self.test_labels = test_labels

        return labels_cnt

    def calculate(self, test_data):
        if self.kmeans == None:
            self.__clustering()
        Y2 = self.__calc_test_proportion(test_data)
        Z = NDB.Ztest(self.train_cnts,
                      self.whitened.shape[0],
                      Y2,
                      test_data.shape[0],
                      self.t_alpha)

        return Z.sum() / self.k_clusters, self.k_clusters

    def plot_hist(self, list_datas=[]):
        SE_train = NDB.StandardErr(self.train_cnts, self.whitened.shape[0],
                                   self.train_cnts, self.whitened.shape[0])

        plt.figure(figsize=(18, 9))

        xs = np.arange(1, self.k_clusters + 1)
        train_ys = self.train_cnts / self.whitened.shape[0]
        train_dys = 2 * SE_train

        train_ys = train_ys[self.bin_order]
        train_dys = train_dys[self.bin_order]

        plt.bar(xs, height=train_dys, bottom=train_ys - SE_train, color='gray', width=1.0, label='Train')

        for data_leg in list_datas:
            test_ys = self.__calc_test_proportion(data_leg[0]) / data_leg[0].shape[0]
            test_ys = test_ys[self.bin_order]

            plt.plot(xs, test_ys, '--*', label=data_leg[1])

        plt.legend(loc='best')
        plt.ylim(0.0, np.max(train_ys) * 1.3)
        plt.show()

    def visualize_min_max_bins(self, test_data, image_size=[64,64], num_channels=3, id=0, mode='max'):
        if mode == 'max':
            max_label = self.max_labels[id]
            cluster_mean = self.kmeans.cluster_centers_[max_label].reshape(1, -1)
            bin_pics = test_data[self.test_labels == max_label][:24]

            bin_pics = np.concatenate([cluster_mean, bin_pics], axis=0)
            bin_pics = from_torch_to_numpy(bin_pics, image_size, num_channels)

            figures = {}

            for i, pic in enumerate(bin_pics):
                if i == 0:
                    figures["mean"] = pic
                else:
                    figures[i] = pic

            plot_figures(figures, 5, 5, (18, 9))
        else:
            min_label = self.min_labels[id]
            cluster_mean = self.kmeans.cluster_centers_[min_label].reshape(1, -1)
            bin_pics = test_data[self.test_labels == min_label][:1]

            bin_pics = np.concatenate([cluster_mean, bin_pics], axis=0)
            bin_pics = from_torch_to_numpy(bin_pics, image_size, num_channels)

            figures = {}

            for i, pic in enumerate(bin_pics):
                if i == 0:
                    figures["mean"] = pic
                else:
                    figures[i] = pic

            plot_figures(figures, 1, 1, (18, 4))


    def __whitening(self, data, eps=1e-3):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + eps
        return (data - mean) / std, mean, std

    @staticmethod
    def StandardErr(y1, n1, y2, n2):
        p = (y1 + y2) / (n1 + n2)
        return np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    @staticmethod
    def Ztest(y1, n1, y2, n2, t_alpha):
        z = (y1 / n1 - y2 / n2) / NDB.StandardErr(y1, n1, y2, n2)
        return z > t_alpha
