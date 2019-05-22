# global imports
import os
import pickle
import datetime
import random
import glob
from collections import defaultdict
import warnings

# INSTALLED PACKAGES
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler  # StandardScaler
from google.cloud import storage
from google.cloud import bigquery

warnings.filterwarnings('ignore') # supress warnings to print out
random.seed(6372675) # set random seed

class DataProcessing(object):
    def __init__(self, key=None, bucket_name=None, save_dir=None, id_col=None, fillna=-1,label_encode_threshold=100, features=[], scalar=None, label_encode_dict={},params=None):
        print('[DataProcessing]: initialize DataProcessing module.')
        # create connection to google cloud platform using service account path as key
        if (key != None) and (bucket_name != None):
            self.local_set_gcs(key, bucket_name)
        elif (key == None) and (bucket_name != None):
            self.cloud_set_gcs(bucket_name)
        else:
            self.bucket = None
        self.train_flag = True # if True that means model needs to be trained
        self.id_col = id_col
        self.fillna = fillna
        self._label_encode_threshold = label_encode_threshold
        self.features = features
        self._scalar = scalar
        self._label_encode_dict = label_encode_dict
        self.params = {}
        if isinstance(params, dict):
            self.id_col = params['id_col']
            self.fillna = params['fillna']
            self._label_encode_threshold = params['label_encode_threshold']
            self.features = params['features']
            self._scalar = params['scalar']
            self._label_encode_dict = params['label_encode_dict']
            self.params = params
        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(self.save_dir+'/pipeline'): # create model directory if it does not already exist
                os.makedirs(self.save_dir+'/pipeline')
            if not os.path.exists(self.save_dir+'/data'): # create model directory if it does not already exist
                os.makedirs(self.save_dir+'/data')
        else:
            self.save_dir = None


    ###################################
    # Data Type Check and Set Methods #
    ###################################
    def _check_data(self, data):
        if isinstance(data, pd.DataFrame):
            return True
        else:
            return False

    def local_set_gcs(self, key, bucket_name):
        # self.project_id = project_id
        storage_client = storage.Client.from_service_account_json(key)
        b = storage_client.bucket(bucket_name)
        if b.exists():
            self.bucket=b
        else:
            print('[DataProcessing:set_gcs]: bucket does not exist.')
        pass

    def cloud_set_gcs(self, bucket_name):
        # self.project_id = project_id
        storage_client = storage.Client()
        b = storage_client.bucket(bucket_name)
        if b.exists():
            self.bucket=b
        else:
            print('[ClusterModel:set_gcs]: bucket does not exist.')
        pass

    ##################################
    ###### Loading/Reading Data ######
    ##################################
    def read_data_from_file(self, filename, sep=None):
        if '.csv' in filename: # check for file type csv
            if sep: # check for custom separator
                print(
                    '[DataProcessing:read_data_from_file]: reading .csv data using custom seperator ({}).'.format(sep))
                return pd.read_csv(filename, sep=sep)
            else:  # use default separator
                print(
                    '[DataProcessing:read_data_from_file]: reading .csv data using default separator (,).')
                self.df = pd.read_csv(filename)
        elif '.xlsx' in filename: # check for file type excel
            print(
                '[DataProcessing:read_data_from_file]: reading .xlsx data.')
            return pd.read_excel(filename)
        elif '.json' in filename: # check for file type .json
            print(
                '[DataProcessing:read_data_from_file]: reading .json data.')
            return pd.read_json(filename)
        else:
            print('[DataProcessing:read_data_from_file]: data should be .csv, .xlsx, .json format.')

    def read_data_gcs(self, filepath=None, key=None, bucket_name=None):
        if self.bucket == None:
            if (key != None) and (bucket_name != None):
                self.local_set_gcs(key, bucket_name)
            elif (key == None) and (bucket_name != None):
                self.cloud_set_gcs(bucket_name)
        if self.bucket.exists():
            # Path to the data inside the public bucket
            print('[DataProcessing:read_data_gcs]: reading data from {} to {}.'.format(self.bucket, self.save_dir))
            if filepath:
                for f in filepath:
                    blob = self.bucket.blob(f)
                    if blob.exists():
                        if not os.path.exists(self.save_dir+'/data'):
                            os.makedirs(self.save_dir+'/data')
                        print('\t', blob)
                        # Download the files
                        blob.download_to_filename(self.save_dir +'/'+'data'+'/'+ f.split('/')[-1])
                        # [END download-data]
                    else:
                        print('[DataProcessing:read_data_gcs]: file {} does not exist.'.format(filepath))
                        pass
            # if data_dir:
            #     blobs=self.bucket.list_blobs(prefix=data_dir, delimiter='/')
            #     for blob in blobs:
            #         print(blob.name)
            #         blob.download_to_filename(self.save_dir +'/'+'data'+'/'+ blob.name)
            df = self.load_data_local_dir(filepath=self.save_dir +'/data/*.csv')
            return df  
        else:
            print('[DataProcessing:read_data_gcs]: provide key and bucket_name.')
            pass

    def load_data_local_dir(self, filepath):
        all_files = glob.glob(filepath)
        dfs = [pd.read_csv(f, index_col=None, header=0) for f in all_files]
        if len(dfs) == 0:
            print('[DataProcessing:load_data_local_dir]: no files in file path')
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df  
        
    def read_data_bq(self, query, key=None):
        print('[DataProcessing:read_data_bq]: reading data from BQ query into pandas DataFrame.')
        print('------QUERY------\n', query)
        print('-----------------')
        # to train locally
        if key != None:
            try:
                client = bigquery.Client.from_service_account_json(key)
            except:
                client = bigquery.Client()
        else:
            client = bigquery.Client() # for train on GCP
        # Perform a query.
        query_job = client.query(query)  # run query
        return query_job.to_dataframe() # get data into pandas df

    def load_data(self, data):
        if isinstance(data, pd.DataFrame): # check for data type pd.DataFrame
            return data
        elif isinstance(data, np.ndarray):  # check for data type np.array
            return pd.DataFrame(data) # read in np.array as pd.DataFrame
        else:
            print('[DataProcessing:load_data]: data should be pd.DataFrame or np.array.')
            return None

    ##################################
    ######### Processing Data ########
    ##################################
    def _onehot_encode(self, df, threshold):
        temp = df.copy()
        # find non-numeric column dtypes in the data
        bool_df = temp.select_dtypes(include=['bool'])
        non_numerical_df = temp.select_dtypes(include=['object'])
        # Turn all boolean into type string
        bool_df = bool_df.astype(str)
        non_numerical_df = pd.concat([non_numerical_df, bool_df], axis=1)
        for c in non_numerical_df.columns:
            # if number of unique values is greater than the threshold it won't dummy the variable
            if non_numerical_df[c].nunique() > threshold:
                # remove col if nunique > threshold value won't dummy the value
                non_numerical_df.drop(c, axis=1, inplace=True)
        # want to keep split column
        if 'split' in non_numerical_df.columns:
            non_numerical_df.drop('split', axis=1, inplace=True)
        # make sure that there are columns in the dataframe to dummy
        if non_numerical_df.shape[1] == 0:
            return temp.loc[:, ~temp.columns.duplicated()]
        else:
            print('[DataProcessing:_onehot_encode]: one hot encoding {} categorical variable(s) for categorical variable(s) with <= {} unique values.'.format(
                non_numerical_df.shape[1], threshold))
            # remove columns to be dummied from data
            temp.drop(non_numerical_df.columns, axis=1, inplace=True)
            # create dummy variable columns for non-numeric data
            dummies_df = pd.get_dummies(
                non_numerical_df, drop_first=False, dummy_na=False)
            false_cols = dummies_df.filter(regex='_False$').columns.tolist()
            dummies_df.drop(false_cols, axis=1, inplace=True)
            # concat dummy columns to data (that has had the original columns removed)
            temp = pd.concat([temp, dummies_df], axis=1)
            return temp.loc[:, ~temp.columns.duplicated()]

    def _label_encode(self, df):
        # label encode categorical variables
        temp = df.copy()
        non_numerical_df = temp.select_dtypes(include=['object', 'bool'])
        non_numerical_df = non_numerical_df.astype(str)
        # want to keep split column
        if 'split' in non_numerical_df.columns:
            non_numerical_df.drop('split', axis=1, inplace=True)
        temp.drop(non_numerical_df.columns, axis=1, inplace=True)
        if non_numerical_df.shape[1] == 0:
            print('[DataProcessing:_label_encode]: no categorical variables to label encode.')
            return temp.loc[:, ~temp.columns.duplicated()]
        else:
            if isinstance(self._label_encode_dict, dict) and self.train_flag==False:
                print('[ClusterModel:_label_encode]: fitting label encoder.')
                # Encoding the variable
                label_encoded_df = non_numerical_df.apply(lambda x: self._label_encode_dict[x.name].transform(x))
                # Inverse the encoded
                # label_encoded_df.apply(lambda x: d[x.name].inverse_transform(x))
                return pd.concat([temp, label_encoded_df], axis=1)
            elif self.train_flag==True:
                print('[DataProcessing:_label_encode]: creating label encoder dict.')
                self._label_encode_dict = defaultdict(LabelEncoder)
                # Encoding the variable
                label_encoded_df = non_numerical_df.apply(
                    lambda x: self._label_encode_dict[x.name].fit_transform(x))
                # Inverse the encoded
                # label_encoded_df.apply(lambda x: d[x.name].inverse_transform(x))
                if self.save_dir:
                    pickle.dump(self._label_encode_dict, open(self.save_dir+'/pipeline/label_encode_dict.pkl', 'wb'))
                return pd.concat([temp, label_encoded_df], axis=1)
            else:
                return temp.loc[:, ~temp.columns.duplicated()]

    def _covert_to_numeric(self, df):
        # convert dtype to float and downcast if possible
        # handles mixed datatypes
        df = df.apply(pd.to_numeric, errors='ignore', downcast='float')
        return df

    def process_data(self, data, train_flag, id_col=None, features=None, fillna=None, scalar=None, threshold=None):
        self.train_flag = train_flag
        print('[DataProcessing:process_data]: processing data for clustering model.')
        if self.train_flag:
            if id_col == None:
                self.id_col=self.id_col
            else:
                self.id_col = id_col
            if fillna == None:
                self.fillna=self.fillna
            else:
                self.fillna = fillna
            if threshold == None:
                self._label_encode_threshold = self._label_encode_threshold
            else:
                self._label_encode_threshold = threshold
        if self.id_col != None and self.id_col in data.columns:
            ids = data[self.id_col]
            # print(data[self.id_col].nunique())
        if self._check_data(data):
            if isinstance(features, list) and len(features) > 0:
                data = data[features]
            # try and turn all possible numeric cols to float
            data = self._covert_to_numeric(data)
            # One-hot encode categorical variables first based on unique value threshold
            data = self._onehot_encode(df=data, threshold=self._label_encode_threshold)
            # Label Encoded any remaining categorical variables
            data = self._label_encode(df=data)
            if self.train_flag:
                features = data.columns.tolist()
                if self.id_col != None:
                    features.remove(self.id_col)
                self.features = features
            elif not self.train_flag and len(self.features) == 0:
                print('[DataProcessing:process_data]: features not passed as attribute using all columns in dataset.')
                self.features = data.columns.tolist()
            # fill nan values
            data.fillna(self.fillna, inplace=True)
            # Scale/normalize data
            
            if self.train_flag:
                print('[DataProcessing:process_data]: creating scalar and scaling data.')
                if scalar:
                    self._scalar = scalar
                else:
                    self._scalar = StandardScaler()# create new scalar if self._scalar does not exist, define StandardScaler and set feature range
                self._scalar = self._scalar.fit(data[self.features]) # fit scalar
                print('[DataProcessing:process_data]: saving scalar.')
                self.save_params_dict()
                encoded_data = self._scalar.transform(data[self.features]) # transform the data
                encoded_data = pd.DataFrame(data=encoded_data, columns=self.features)
                if self.id_col != None and self.id_col in data.columns:
                    # print(encoded_data.shape, data.shape, ids.shape)
                    encoded_data = pd.concat([ids.reset_index(), encoded_data], axis=1)
                    encoded_data.drop('index', axis=1, inplace=True)
                return encoded_data, self.features
            if not self.train_flag:
                if not self._scalar:
                    self._scalar = pickle.load(open('/scalar.pkl', 'rb'))
                print('[DataProcessing:process_data]: scaling data with trained scalar.')
                # if scalar exists use to transform data
                encoded_data = self._scalar.transform(data[self.features])
                encoded_data = pd.DataFrame(data=encoded_data, columns=self.features)
                if self.id_col != None and self.id_col in data.columns:
                    # print(self.id_col)
                    # print(encoded_data.shape, data.shape, data[self.id_col].shape)
                    encoded_data = pd.concat([data[self.id_col].reset_index(), encoded_data], axis=1)
                    encoded_data.drop('index', axis=1, inplace=True)
                return encoded_data
        else:
            print('[DataProcessing:process_data]: data is not in pd.DataFrame format.')
            return

    def save_params_dict(self):
        self.params['fillna'] = self.fillna
        self.params['id_col'] = self.id_col
        self.params['label_encode_threshold'] = self._label_encode_threshold
        self.params['features'] = self.features
        self.params['scalar'] = self._scalar
        self.params['label_encode_dict'] = self._label_encode_dict
        if self.save_dir:
            pickle.dump(self.params, open(self.save_dir+'/pipeline/dp_params.pkl', 'wb'))

    # def save_pipeline(self):
    #     self.bucket=None
    #     print('[DataProcessing:save_pipeline]: saving DataProcessing pipeline to {}.'.format(self.save_dir+'/pipeline'))
    #     pickle.dump(self, open(self.save_dir+'/pipeline/dp_pipeline.pkl', 'wb'))