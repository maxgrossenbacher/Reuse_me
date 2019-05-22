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

warnings.filterwarnings('ignore') # supress warnings to print out
random.seed(1465297) # set random seed


class ClusterModel(object):
    def __init__(self, model_folder=None, filename=None, sep=None, id_col=None, key=None, bucket_name=None):
        print('[ClusterModel]: initialize ClusterModel module')
        # create connection to google cloud platform using service account path as key
        if (key != None) and (bucket_name != None):
            self.local_set_gcs(key, bucket_name)
        elif (key == None) and (bucket_name != None):
            self.cloud_set_gcs(bucket_name)
        else:
            self.bucket = None
        # set id_col
        self.id_col = id_col
        self._date = datetime.datetime.today().strftime('%Y%m%d_%H%M%S') # set global date
        self.centroids = None
        self.error = None
        self.s_score = None
        self.cluster_model = None
        self.features = None
        if model_folder:
            self.model_folder = model_folder
            if not os.path.exists(self.model_folder): # create model directory if it does not already exist
                os.makedirs(self.model_folder)
        else:
            self.model_folder = None
        

    ##################################
    ###### Loading/Reading Data ######
    ##################################
    def read_data_from_file(self, filename, sep=None):
        if '.csv' in filename: # check for file type csv
            if sep: # check for custom separator
                print(
                    '[ClusterModel:read_data_from_file]: reading .csv data using custom seperator ({}).'.format(sep))
                self.df = pd.read_csv(filename, sep=sep)
            else:  # use default separator
                print(
                    '[ClusterModel:read_data_from_file]: reading .csv data using default separator (,).')
                self.df = pd.read_csv(filename)
        elif '.xlsx' in filename: # check for file type excel
            print(
                '[ClusterModel:read_data_from_file]: reading .xlsx data.')
            self.df = pd.read_excel(filename)
        elif '.json' in filename: # check for file type .json
            print(
                '[ClusterModel:read_data_from_file]: reading .json data.')
            self.df = pd.read_json(filename)
        else:
            print('[ClusterModel:read_data_from_file]: data should be .csv, .xlsx, .json format.')
        
    def load_data(self, data):
        if isinstance(data, pd.DataFrame): # check for data type pd.DataFrame
            self.df = data
        elif isinstance(data, np.ndarray):  # check for data type np.array
            self.df = pd.DataFrame(data) # read in np.array as pd.DataFrame
        else:
            print('[ClusterModel:load_data]: data should be pd.DataFrame or np.array.')
            self.df = None
    
    def _load_testing_data(self, data):
        if isinstance(data, pd.DataFrame):  # check for data type pd.DataFrame
            df = data
        elif isinstance(data, np.ndarray):  # check for data type np.array
            df = pd.DataFrame(data)  # read in np.array as pd.DataFrame
        else:
            print('[ClusterModel:load_data]: data should be pd.DataFrame or np.array.')
            df = None
        return df

    ###################################
    # Data Type Check and Set Methods #
    ###################################
    def _check_data(self):
        if isinstance(self.df, pd.DataFrame):
            return True
        else:
            return False
    
    def _check_model(self):
        if isinstance(self.cluster_model, KMeans):
            return True
        else:
            return False

    def set_id_col(self, id_col):
        if self.id_col == None:
            self.id_col = id_col
        else:
            print('[ClusterModel:set_id_col]: id_col has already been set to {}.'.format(self.id_col))
        pass

    def reset_id_col(self, id_col):
        self.id_col = id_col
        print('[ClusterModel:reset_id_col]: reset id_col to {}.'.format(
            self.id_col))
        pass

    def set_model_folder(self, model_folder):
        if self.model_folder == None:
            self.model_folder = model_folder
            if not os.path.exists(self.model_folder):
                    os.makedirs(self.model_folder)
        else:
            print('[ClusterModel:set_model_folder]: model folder has already been set to {}.'.format(self.model_folder))
        pass

    def reset_model_folder(self, model_folder):
        self.model_folder = model_folder
        if not os.path.exists(self.model_folder):
                os.makedirs(self.model_folder)
        print('[ClusterModel:reset_model_folder]: reset model folder to {}.'.format(
            self.model_folder))
        pass

    def local_set_gcs(self, key, bucket_name):
        # self.project_id = project_id
        storage_client = storage.Client.from_service_account_json(key)
        b = storage_client.bucket(bucket_name)
        if b.exists():
            print('[ClusterModel:local_set_gcs]: set bucket {}.'.format(b))
            self.bucket=b
        else:
            print('[ClusterModel:local_set_gcs]: bucket does not exist.')
        pass

    def cloud_set_gcs(self, bucket_name):
        # self.project_id = project_id
        storage_client = storage.Client()
        b = storage_client.bucket(bucket_name)
        if b.exists():
            print('[ClusterModel:cloud_set_gcs]: set bucket {}.'.format(b))
            self.bucket=b
        else:
            print('[ClusterModel:cloud_set_gcs]: bucket does not exist.')
        pass

    ##################################
    ######### Model Training #########
    ##################################
    def get_centroids(self, model=None):
        if model != None:
            self.cluster_model=model
        if self._check_model():
            return self.cluster_model.cluster_centers_
        else:
            print('[ClusterModel:centroids]: no model trained.')

    def get_error(self, model=None):
        if model != None:
            self.cluster_model=model
        if self._check_model():
            return self.cluster_model.inertia_
        else:
            print('[ClusterModel:get_error]: no model trained.')
    
    def get_silhouette_score(self, model=None):
        if model != None:
            self.cluster_model=model
        if self._check_model():
            return silhouette_score(self.df[self.features].values, self.cluster_model.labels_, metric='euclidean')
        else:
            print('[ClusterModel:get_silhouette_score]: no model trained.')

    def train(self, clusters, features=None, id_col=None, save=True, key=None, bucket_name=None):
        self.clusters = clusters
        if isinstance(features,list):
            self.features = features
        else:
            self.features = self.df.columns.tolist()
        self.set_id_col(id_col)
        if self.id_col != None:
            if self.id_col in self.features:
                self.features.remove(self.id_col)
        print('[ClusterModel:train]: training {} clusters on {} samples using {} features.' .format(self.clusters,
            self.df.shape[0], len(self.features)))
        # train model
        # print('define model')
        self.cluster_model = KMeans(n_clusters=self.clusters, n_init=25,
                            max_iter=1000, n_jobs=-1, random_state=40)
        # print('train model')
        self.cluster_model = self.cluster_model.fit(self.df[self.features].values)
        # print('Done.')
        if save:
            if self.model_folder:
                if not os.path.exists(self.model_folder+'/model_{}'.format(self.clusters)):
                    os.makedirs(self.model_folder+'/model_{}'.format(self.clusters))
                print('[ClusterModel:train]: saving model.')
                pickle.dump(self.cluster_model, open(self.model_folder + '/model_{}/model.pkl'.format(self.clusters), 'wb'))
                self.create_model_report()
            else:
                print('[ClusterModel:train]: please provide model folder use set_model_folder method.')
        return self.cluster_model

    def iter_train(self, cluster_list, features=None, id_col=None, save=True, key=None, bucket_name=None):
        if isinstance(cluster_list, list):
            model_dict = {}
            print('[ClusterModel:iter_train]: iterating through cluster_list.')
            for c in tqdm(cluster_list):
                model = self.train(int(c), features=features, id_col=id_col, save=save, key=key, bucket_name=bucket_name)
                model_dict[int(c)] = model
            return OrderedDict(sorted(model_dict.items())) # sorted by key ordered dict
        else:
            print('[ClusterModel:iter_train]: pass a list of integers to cluster_list.')
            return

    ##################################
    ######## Model Evaluation ########
    ##################################
    def model_selector(self, model_dict, plot=False, save=True): # code for selecting optimal K-means model
        # code for selecting optimal cluster model returns the model
        errors = []
        model_dict = OrderedDict(sorted(model_dict.items()))
        cluster_list = list(model_dict.keys())
        for m in model_dict.values():
            error = self.get_error(m)
            errors.append(error)
        # Calculate elbow
        kn = KneeLocator(cluster_list, errors, direction='decreasing', curve='convex')
        elbow = kn.knee
        print('[ClusterModel:model_selector]: recommended K = {}'.format(elbow))
        # Plot Elbow
        fig, ax = plt.subplots(1)
        ax.plot(cluster_list, errors, color='b', marker='o')
        ax.vlines(x=elbow, linestyles='--', color='black', ymin=0, ymax=max(errors)+10)
        ax.set_title('Cluster Model Elbow Plot - Optimal K = {}'.format(elbow))
        ax.set_xlabel('Cluster K')
        ax.set_ylabel('Model Error (Inertia)')
        if save:
            if self.model_folder:
                if not os.path.exists(self.model_folder+'/results'):
                    os.makedirs(self.model_folder+'/results')
                print('[ClusterModel:model_selector]: saving Elbow Plot to {} dir.'.format(self.model_folder +'/results/'))
                plt.savefig(self.model_folder +'/results/Elbow_Plot.png')
        if plot:
            plt.show()
        return model_dict[elbow]

    def predict(self, model, data, save=True, key=None, bucket_name=None):
        if self._check_model:
            data = self._load_testing_data(data)
            print('[ClusterModel:predict]: processing and predicting clusters on {} samples.'.format(data.shape[0]))
            cluster_preds = model.predict(data[self.features].values)
            cluster_preds_df = pd.DataFrame({'cluster': cluster_preds})
            if self.id_col != None:
                ids = data[self.id_col]
                results = pd.concat([ids.reset_index(), cluster_preds_df], axis=1)
                results.drop('index', axis=1, inplace=True)
            else:
                results = cluster_preds_df
            if save:
                if self.model_folder:
                    if not os.path.exists(self.model_folder+'/results'):
                        os.makedirs(self.model_folder+'/results')
                    print(
                        '[ClusterModel:predict]: saving new cluster analysis results.')
                    results.to_csv(self.model_folder +
                                   '/results/clustering_results.csv', index=False)
                else:
                    print(
                        '[ClusterModel:predict]: please provide model folder use set_model_folder method.')
            return results
        else:
            print('[ClusterModel:predict]: model is not scikit-learn K-means clustering model.')
            return

    def cluster_evaluation(self, predictions, data, save=True, key=None, bucket_name=None):
        if self._check_model:
            # print(data.columns)
            print('[ClusterModel:cluster_evaluation]: evaluating clusters of {} samples.'.format(len(predictions)))
            if self.id_col in data.columns:
                eval_results = pd.merge(predictions, data, on=self.id_col, how='inner')
                eval_results.drop_duplicates([self.id_col], keep='first', inplace=True)
            else:
                eval_results = pd.concat([predictions, data.reset_index()], axis=1)
            if self.id_col != None:
                cluster_count = eval_results.groupby('cluster')[self.id_col].count().reset_index()
                cluster_count.rename(columns={self.id_col:'cluster_count'}, inplace=True)
            else:
                col = eval_results.columns.tolist()[0]
                cluster_count = eval_results.groupby('cluster')[col].count().reset_index()
                cluster_count.rename(columns={col:'cluster_count'}, inplace=True)
            summary_results = eval_results.groupby('cluster').mean().reset_index()
            summary_results = summary_results.merge(cluster_count, on='cluster')
            summary_results['cluster_precentage'] = cluster_count['cluster_count'] / \
                cluster_count['cluster_count'].sum()
            if save:
                if self.model_folder:
                    if not os.path.exists(self.model_folder+'/results'):
                        os.makedirs(self.model_folder+'/results')
                    print(
                        '[ClusterModel:cluster_evaluation]: saving new cluster analysis results.')
                    summary_results.to_csv(self.model_folder +
                                   '/results/eval_report_summary.csv', index=False)
                    eval_results.to_csv(self.model_folder +
                                        '/results/eval_report_results.csv', index=False)
                else:
                    print(
                        '[ClusterModel:predict]: please provide model folder use set_model_folder method.')
            return eval_results
        else:
            print('[ClusterModel:predict]: no model trained.')
            return

    def create_model_report(self):
        print('[ClusterModel:create_model_report]: creating model report.')
        report = pd.DataFrame({'model_bucket':[self.bucket],
                        'cloud_dir':['model_'+self._date],
                        'model_clusters':[self.clusters],
                        'num_training_features':[len(self.features)],
                        'num_training_samples':[self.df.shape[0]],
                        'id_col':[self.id_col],
                        'model_inertia':[self.get_error()],
                        # 'sihouette_score': [self.get_silhouette_score()]
                        })
        if not os.path.exists(self.model_folder+'/results'):
            os.makedirs(self.model_folder+'/results')
        report.to_csv(self.model_folder+'/results/model_report.csv')
        pass
    
    ##################################
    ####### Moving Model to GCS ######
    ##################################
    def move_to_gcs(self, dirs=['model', 'results', 'data', 'pipeline'], bucket_name=None, key=None,):
        if (bucket_name and key):
            self.local_set_gcs(key, bucket_name)
        elif bucket_name:
            self.cloud_set_gcs(bucket_name)
        if self.bucket.exists():
            # saving summary results
            for d in dirs:
                fs = glob.glob(self.model_folder+'/'+d+'/*')
                for f in fs:
                    blob = self.bucket.blob(
                    '{}/{}/{}'.format(('model_'+self._date), d, f.split('/')[-1]))
                    print('[ClusterModel:move_to_gcs]: moving file: {} to {}.' .format(f, blob))
                    blob.upload_from_filename(f)
            print('[ClusterModel:move_to_gcs]: Done.')
        pass
