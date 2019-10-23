import numpy as np
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

#images should be flatten and ndarray type.
#Supported only for the whole datasets
class NDB():
    def __init__(self, train_data, k_clusters, alpha=0.05, eps=1e-3, verbose=False):
        self.whitened, self.mean, self.std = self.__whitening(train_data, eps=eps)
        
        self.k_clusters = k_clusters
        self.t_alpha = norm.ppf(1 - alpha)
        self.verbose = verbose
        self.kmeans = None
        self.train_cnts = None
        self.order = None
        
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

        if self.verbose:
            print('Clustered into %d clusters' % self.k_clusters)

        labels_cnt = np.zeros(self.k_clusters)
        labels_cnt[label_vals] = label_counts
        self.train_cnts = labels_cnt
        pass

    def __calc_test_proportion(self, test_data):
        whitened = (test_data - self.mean)/self.std
        labels_cnt = np.zeros(self.k_clusters)
        test_labels = self.kmeans.predict(whitened)
        label_vals, label_counts = np.unique(test_labels, return_counts=True)
        labels_cnt[label_vals] = label_counts
        
        bin_order = np.argsort(-label_counts)
        
        self.min_labels = label_vals[bin_order][-5:]
        self.max_labels = label_vals[bin_order][:5]
        
        self.test_labels = test_labels
        
        return labels_cnt
    
    def vizaulize(self, test_data, init_data_shape, mode='max', n_pic=5):
        
        l_set = self.max_labels if mode == 'max' else self.min_labels
        for l in l_set:
          data_viz = test_data[self.test_labels == l][:n_pic]#, mnist_NDB.min_labels
          plt.figure(figsize=(4*n_pic + 1, 4))
          for i in range(len(data_viz)):
            plt.subplot(1, n_pic, i + 1)
            plt.axis('off')
            if len(init_data_shape) == 2:
              plt.imshow(data_viz[i].reshape(init_data_shape))#.transpose(0, 2))
            elif len(init_data_shape) == 3:
              plt.imshow(data_viz[i].reshape(init_data_shape).transpose(0, 2))
          plt.show()
          
       
    
    def calculate(self, test_data):
        if self.kmeans == None:
            self.__clustering()
        Y2 = self.__calc_test_proportion(test_data)
        Z = NDB.Ztest(self.train_cnts,
                      self.whitened.shape[0],
                      Y2,
                      test_data.shape[0], 
                      self.t_alpha)
        
        return Z.sum()/self.k_clusters, self.k_clusters

    def plot_hist(self, list_datas=[]):
        SE_train = NDB.StandardErr(self.train_cnts, self.whitened.shape[0], 
                                  self.train_cnts, self.whitened.shape[0])
        
        order = np.argsort(-self.train_cnts)
        
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(1, self.k_clusters + 1), 
                self.train_cnts[order]/self.whitened.shape[0], 
                yerr=2*SE_train[order], alpha=0.5, label='Train')
        
        for data_leg in list_datas:
            y = self.__calc_test_proportion(data_leg[0])
            plt.plot(np.arange(1, self.k_clusters + 1), y[order]/data_leg[0].shape[0], label=data_leg[1])
        plt.legend(loc='best')
        plt.show()
      

    def __whitening(self, data, eps=1e-3):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + eps
        return (data - mean)/std, mean, std
        
    @staticmethod
    def StandardErr(y1, n1, y2, n2):
        p = (y1 + y2)/(n1 + n2)
        return np.sqrt(p*(1 - p)*(1/n1 + 1/n2))

    @staticmethod
    def Ztest(y1, n1, y2, n2, t_alpha):
        z = (y1/n1 - y2/n2)/NDB.StandardErr(y1, n1, y2, n2)
        return z > t_alpha