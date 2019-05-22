# global imports
import os
import pickle
import datetime
import random
import glob
from collections import defaultdict, OrderedDict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore') # supress warnings to print out

# INSTALLED PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from google.cloud import storage
from kneed.knee_locator import KneeLocator
from cluster_class import ClusterModel
from x_means import XMeans

warnings.filterwarnings('ignore') # supress warnings to print out
random.seed(1465297) # set random seed

class XMeansModel(ClusterModel):
    def train(self, kmax=50, features=None, id_col=None, save=True, key=None, bucket_name=None):
        if isinstance(features,list):
            self.features = features
        else:
            self.features = self.df.columns.tolist()
        self.set_id_col(id_col)
        if self.id_col != None:
            if self.id_col in self.features:
                self.features.remove(self.id_col)
        print('[XMeansModel:train]: optimizing number of clusters on {} samples using {} features.' .format(
            self.df.shape[0], len(self.features)))
        self.cluster_model = XMeans(kmax=kmax, max_iter=100, n_jobs=-1, random_state=40)
        self.cluster_model = self.cluster_model.fit(self.df[self.features].values)
        self.clusters = self.cluster_model.n_clusters
        print('[XMeansModel:train]: XMeans algorithm finds the k = {} is optimal.'.format(self.clusters))
        if save:
            if self.model_folder:
                if not os.path.exists(self.model_folder+'/model_{}'.format(self.clusters)):
                    os.makedirs(self.model_folder+'/model_{}'.format(self.clusters))
                print('[XMeansModel:train]:: saving model.')
                pickle.dump(self.cluster_model, open(self.model_folder + '/model_{}/model.pkl'.format(self.clusters), 'wb'))
                self.create_model_report()
            else:
                print('[XMeansModel:train]: please provide model folder use set_model_folder method.')
        return self.cluster_model