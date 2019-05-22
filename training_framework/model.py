import os
import sys
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import datetime
import re
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, precision_score, recall_score, f1_score, accuracy_score
import random


# user defined scripts
from config import MODEL_PARAMS

# module_path = os.path.abspath(os.path.join('../visualization/'))
# print(module_path)
# if module_path not in sys.path:
#     sys.path.append(module_path)
# from visualize import actual_v_predictions_plot, global_feature_importance
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

random.seed(2394)


class Model(object):
    def __init__(self, model_folder ='model' , model_params =MODEL_PARAMS, model_name=None):
        print('[model]: initialize model training')
        self.model_name = model_name
        self.model_params =model_params
        self.model_folder =model_folder
        self.target =None
        self.data =None # of type Features object (from build_features.py)
        self.features =None
        self.model =None
        self.model_results ={'train':{}, 'val':{}, 'test':{}}
        self.validated_sets =[]
        self.predictions ={'train':{}, 'val':{}, 'test':{}}
        self.label_encode_dict = {}
        pass

    def _build_model(self):
        if self.model_params:
            print('\t[model]: building model using params in config.py')
            # build regression model according to passed parameters in the config.py file
            self.model =xgb.XGBRegressor(
                                            base_score= np.mean(self.y_train),
                                            **self.model_params
                                            )
        else:
            print('\t[model]: building default model')
            # build default regression model
            self.model =xgb.XGBRegressor(n_estimators=1000000,
                                            base_score= np.mean(self.y_train)
                                        )
        pass

    def load_data(self, data, target):
        if isinstance(data, pd.DataFrame) and target in data.columns:
            self.data =data
            self.target =target
        else:
            print('Data or Target Error: data is not a pd.DataFrame or target not in data columns')

    def load_train_data(self, data, target):
        if isinstance(data, pd.DataFrame) and target in data.columns:
            self.y_train = data.pop(target)
            self.X_train = data
            self.features = self.X_train.columns
        else:
            print(
                'Data or Target Error: data is not a pd.DataFrame or target not in data columns')
    
    def load_validation_data(self, data, target):
        if isinstance(data, pd.DataFrame) and target in data.columns:
            self.y_val = data.pop(target)
            self.X_val = data
        else:
            print(
                'Data or Target Error: data is not a pd.DataFrame or target not in data columns')

    def load_test_data(self, data, target):
        if isinstance(data, pd.DataFrame) and target in data.columns:
            self.y_test = data.pop(target)
            self.X_test = data
        else:
            print(
                'Data or Target Error: data is not a pd.DataFrame or target not in data columns')

    def _check_for_model(self):
        if isinstance(self.model, xgb.sklearn.XGBRegressor):
            return True
        else:
            return False

    def _check_for_predictions(self):
        if isinstance(self.predictions['test'], np.ndarray):
            self.validated_sets.append('test')
        if isinstance(self.predictions['val'], np.ndarray):
            self.validated_sets.append('val')
        if isinstance(self.predictions['train'], np.ndarray):
            self.validated_sets.append('train')
        pass

    def _onehot_encode(self, df, threshold=10):
        temp = df.copy()
        non_numerical_df =temp.select_dtypes(include=['object','bool'])
        non_numerical_df =non_numerical_df.astype(str)
        for c in non_numerical_df.columns:
            if non_numerical_df[c].nunique() > threshold:
                non_numerical_df.drop(c, axis=1, inplace=True)
        # want to keep split column
        if 'split' in non_numerical_df.columns:
            non_numerical_df.drop('split', axis=1, inplace=True)
        if non_numerical_df.shape[1] == 0:
            return temp.loc[:,~temp.columns.duplicated()]
        else:
            temp.drop(non_numerical_df.columns, axis=1, inplace=True)
            dummies_df = pd.get_dummies(non_numerical_df, drop_first=False, dummy_na=False)
            temp = pd.concat([temp, dummies_df], axis=1)
            return temp.loc[:,~temp.columns.duplicated()]

    def _label_encode_categorical_variables(self, df):
        # label encode categorical variables
        temp = df.copy()
        non_numerical_df =temp.select_dtypes(include=['object','bool'])
        non_numerical_df = non_numerical_df.astype(str)
        # want to keep split column
        if 'split' in non_numerical_df.columns:
            non_numerical_df.drop('split', axis=1, inplace=True)
        temp.drop(non_numerical_df.columns, axis=1, inplace=True)
        print('\t[model]: fitting label encoder')
        self.label_encode_dict = defaultdict(LabelEncoder)
        # Encoding the variable
        label_encoded_df = non_numerical_df.apply(lambda x: self.label_encode_dict[x.name].fit_transform(x))
        # Inverse the encoded
        # label_encoded_df.apply(lambda x: d[x.name].inverse_transform(x))
        return pd.concat([temp, label_encoded_df], axis=1)

    def _split(self):
        print('\t[model]: applying train-validation-test split')
        # filter on split column
        train =self.encoded_data[self.encoded_data['split'] =='Train']
        validation =self.encoded_data[self.encoded_data['split'] =='Validation']
        test =self.encoded_data[self.encoded_data['split'] =='Test']

        # pop of the target column for model training and validation
        self.y_train =train.pop(self.target)
        self.y_val =validation.pop(self.target)
        self.y_test =test.pop(self.target)

        # created for model validation
        self._true_values ={'train':self.y_train,
                            'test':self.y_test,
                            'val':self.y_val}

        # remove split column from feature set
        train.drop('split',axis=1, inplace=True)
        validation.drop('split',axis=1, inplace=True)
        test.drop('split',axis=1, inplace=True)

        # define train, validation and test set
        self.X_train =train
        self.X_val =validation
        self.X_test =test
        # get model features
        self.features =self.X_train.columns.tolist()
        pass

    def train(self, eval_metric=['rmse'], drop_cols=[], threshold=0):
        # get date to append to save_model name
        date = datetime.datetime.today().strftime('%Y-%m-%d-%H')

        for c in drop_cols:
            if c in self.data.columns:
                self.data.drop(c, axis=1, inplace=True)
        # One-hot encode
        self.encoded_data =self._onehot_encode(df=self.data, threshold=threshold)
        # Label Encoded
        self.encoded_data =self._label_encode_categorical_variables(df=self.encoded_data)

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        self.encoded_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in self.encoded_data.columns.values]
        # get train-val-test split
        self._split()
        # create xgboost model
        self._build_model()
        # train/fit xgboost model to training set, evaluation on validation set
        print('\t[model]: training model')
        self.model.fit(self.X_train[self.features].values,
                        self.y_train,
                        eval_set =[(self.X_train[self.features].values, self.y_train),(self.X_val[self.features].values, self.y_val)],
                        eval_metric =eval_metric,
                        verbose =10,
                        early_stopping_rounds =100
                        )
        print('\t[model]: saving model and features list')
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if self.model_name:
            pickle.dump(self.features, open(self.model_folder +
                                            '/features_{}.p'.format(self.model_name), 'wb'))
            pickle.dump(self.model, open(self.model_folder +
                                         '/{}'.format(self.model_name), 'wb'))
        else:
            pickle.dump(self.features, open(self.model_folder +
                                            '/features_{}.p'.format(date), 'wb'))
            pickle.dump(self.model, open(self.model_folder +
                                         '/xgboost_model_{}.pkl'.format(date), 'wb'))
        pass

    def simple_train(self, eval_metric=['rmse']):
        # get date to append to save_model name
        date = datetime.datetime.today().strftime('%Y-%m-%d-%H')
        self._build_model()
        # train/fit xgboost model to training set, evaluation on validation set
        print('\t[model]: training model')
        self.model.fit(self.X_train[self.features].values,
                       self.y_train,
                       eval_set=[(self.X_train[self.features].values, self.y_train),
                                 (self.X_val[self.features].values, self.y_val)],
                       eval_metric=eval_metric,
                       verbose=10,
                       early_stopping_rounds=100
                       )
        print('\t[model]: saving model and features list')
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if self.model_name:
            pickle.dump(self.features, open(self.model_folder +
                                            '/features_{}.p'.format(self.model_name), 'wb'))
            pickle.dump(self.model, open(self.model_folder +
                                         '/{}'.format(self.model_name), 'wb'))
        else:
            pickle.dump(self.features, open(self.model_folder +
                                            '/features_{}.p'.format(date), 'wb'))
            pickle.dump(self.model, open(self.model_folder +
                                         '/xgboost_model_{}.pkl'.format(date), 'wb'))
        pass

    def load_model(self, model_filepath, features_filepath):
        print('\t[model]: loading model and feature list')
        self.model = pickle.load(open(model_filepath, 'rb'))
        self.features = pickle.load(open(features_filepath, 'rb'))
        self.model_params = self.model.get_xgb_params()
        pass

    def apply(self, data, probailities=False):
        if self._check_for_model():
            print('\t[model]: applying model')
            preds = self.model.predict(data[self.features].values)
            return preds
        else:
            print('No model found - train or load xgboost model')
        pass

    def validate(self, metrics=['mae','rmse','mape','explained_variance'],set=['train','val','test'], plot=False, labels=None, save=False, fig_size=(5,5), contour=True):
        if self._check_for_model():
            print('\t[model]: applying model')
            for s in set:
                if isinstance(self.X_test[self.features], pd.DataFrame) and s =='test':
                    self.predictions['test'] = self.model.predict(self.X_test[self.features].values)
                if isinstance(self.X_val[self.features], pd.DataFrame) and s =='val':
                    self.predictions['val'] = self.model.predict(self.X_val[self.features].values)
                if isinstance(self.X_train[self.features], pd.DataFrame) and s =='train':
                    self.predictions['train'] = self.model.predict(self.X_train[self.features].values)
        else:
            print('No model found - train or load xgboost model')
        self._check_for_predictions()
        if len(self.validated_sets) > 0:
            print('\t[model]: validating model')
            for s in self.validated_sets:
                for m in metrics:
                    if 'mae':
                        self.model_results[s]['mae'] =mean_absolute_error(self._true_values[s], self.predictions[s])
                    if 'rmse':
                        self.model_results[s]['rmse'] =(mean_squared_error(self._true_values[s], self.predictions[s]))**0.5
                    if 'mape':
                        self.model_results[s]['mape'] =mean_absolute_precentage_error(self._true_values[s], self.predictions[s])
                    if 'explained_variance':
                        self.model_results[s]['explained_variance'] =explained_variance_score(self._true_values[s], self.predictions[s])
                if plot:
                    if s == 'train' and len(labels) > 1:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    label_dummies =self.X_train[labels],
                                                    save =self.model_folder+'/'+save+str('_train'),
                                                    contour = contour
                                                    )
                    if s == 'val' and len(labels) > 1:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    label_dummies =self.X_val[labels],
                                                    save =self.model_folder+'/'+save+str('_val'),
                                                    contour = contour
                                                    )
                    if s == 'test' and len(labels) > 1:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    label_dummies =self.X_test[labels],
                                                    save =self.model_folder+'/'+save+str('_test'),
                                                    contour = contour
                                                    )
                    if s == 'train' and len(labels) == 1:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    labels =self.X_train[labels[0]],
                                                    save =self.model_folder+'/'+save+str('_train'),
                                                    contour = contour
                                                    )
                    if s == 'val' and len(labels) == 1:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    labels =self.X_val[labels[0]],
                                                    save =self.model_folder+'/'+save+str('_val'),
                                                    contour = contour
                                                    )
                    if s == 'test' and len(labels) == 1:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    labels =self.X_test[labels[0]],
                                                    save =self.model_folder+'/'+save+str('_test'),
                                                    contour = contour
                                                    )
                    if not labels:
                        actual_v_predictions_plot(self._true_values[s],
                                                    self.predictions[s],
                                                    title ='Actual v.s. Predicted {} for {} set'.format(self.target, s),
                                                    fig_size =fig_size,
                                                    ci =80,
                                                    color ='orange',
                                                    label_dummies =None,
                                                    save =self.model_folder+'/'+save+'_'+str(s),
                                                    contour = contour
                                                    )
            results_df = pd.DataFrame(self.model_results)
            results_df.to_csv(self.model_folder+'/final_model_results.csv')
                    # actual_v_predictions_plot_contour(self._true_values[s], self.predictions[s],
                                                        # fig_size=(5,5), save=False)
        else:
            ('Prediction Not Found - apply model before validating')
        pass

    def get_global_feature_importance(self, type='weight', n_features=25, return_df=False):
        '''
        type -
            ‘weight’: the number of times a feature is used to split the data across all trees.
            ‘gain’: the average gain across all splits the feature is used in.
            ‘cover’: the average coverage across all splits the feature is used in.
            ‘total_gain’: the total gain across all splits the feature is used in.
            ‘total_cover’: the total coverage across all splits the feature is used in.
        '''
        scores = self.model.get_booster().get_score(importance_type=type)
        mapper = {'f{0}'.format(i): v for i, v in enumerate(self.features)}
        mapped = {mapper[k]: v for k, v in scores.items()}
        col_type = type+'_type'
        df = pd.DataFrame({'Features':list(mapped.keys()), col_type:list(mapped.values())})
        df.sort_values(by=col_type, ascending=False, inplace=True)
        df.reset_index(inplace=True)
        df.to_csv(self.model_folder+'/Global_Feature_Importance.csv')
        # get global feature importance
        global_feature_importance(mapped, type=type, n_features=n_features, save=self.model_folder+'/Global_Feature_Importance')
        if return_df:
            return df
        else:
            pass


class RegressionModel(Model):
    def __init__(self, model_folder ='model' , model_params =MODEL_PARAMS):
        Model.__init__(self, model_folder =model_folder, model_params =model_params)

class ClassificationModel(Model):
    def __init(self, model_folder ='model' , model_params =MODEL_PARAMS):
        Model.__init__(self, model_folder =model_folder, model_params =model_params)

    def _build_model(self):
        if self.model_params:
            print('\t[model]: building model using params in config.py')
            # build regression model according to passed parameters in the config.py file
            self.model =xgb.XGBClassifier(
                                            **self.model_params
                                            )
        else:
            print('\t[model]: building default model')
            # build default regression model
            self.model =xgb.XGBClassifier(n_estimators=5000, objective='binary:logistic')
        pass

    def _check_for_model(self):
        if isinstance(self.model, xgb.sklearn.XGBClassifier):
            return True
        else:
            return False

    def train(self, eval_metric=['mlogloss'], drop_cols=[], threshold=0):
        # get date to append to save_model name
        date = datetime.datetime.today().strftime('%Y-%m-%d-%H')

        for c in drop_cols:
            if c in self.data.columns:
                self.data.drop(c, axis=1, inplace=True)
        # One-hot encode
        self.encoded_data = self._onehot_encode(
            df=self.data, threshold=threshold)
        # Label Encoded
        self.encoded_data = self._label_encode_categorical_variables(
            df=self.encoded_data)

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        self.encoded_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(
            ('[', ']', '<'))) else col for col in self.encoded_data.columns.values]
        # get train-val-test split
        self._split()
        # create xgboost model
        self._build_model()
        # train/fit xgboost model to training set, evaluation on validation set
        print('\t[model]: training model')
        self.model.fit(self.X_train[self.features].values,
                       self.y_train,
                       eval_set=[(self.X_train[self.features].values, self.y_train),
                                 (self.X_val[self.features].values, self.y_val)],
                       eval_metric=eval_metric,
                       verbose=10,
                       early_stopping_rounds=100
                       )
        print('\t[model]: saving model and features list')
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if self.model_name:
            pickle.dump(self.features, open(self.model_folder +
                                            '/features_{}.p'.format(self.model_name), 'wb'))
            pickle.dump(self.model, open(self.model_folder +
                                         '/{}'.format(self.model_name), 'wb'))
        else:
            pickle.dump(self.features, open(self.model_folder +
                                            '/features_{}.p'.format(date), 'wb'))
            pickle.dump(self.model, open(self.model_folder +
                                         '/xgboost_model_{}.pkl'.format(date), 'wb'))
        pass

    def apply(self, data, probailities=False):
        if self._check_for_model() and probailities == False:
            print('\t[model]: applying model')
            preds = self.model.predict(data[self.features].values)
            return preds
        elif self._check_for_model() and probailities == True:
            print('\t[model]: applying model')
            preds = self.model.predict_proba(data[self.features].values)
            return preds
        else:
            print('No model found - train or load xgboost model')
        pass

    def validate(self, metrics=['precision','recall','f1-score','accuracy'], plot=False):
        self._check_for_predictions()
        if len(self.validated_sets) > 0:
            print('\t[model]: validating model')
            for s in self.validated_sets:
                for m in metrics:
                    if 'precision':
                        self.model_results[s]['precision'] =precision_score(self._true_values[s], self.predictions[s], average='weighted')
                    if 'recall':
                        self.model_results[s]['recall'] =recall_score(self._true_values[s], self.predictions[s], average='weighted')
                    if 'f1-score':
                        self.model_results[s]['f1-score'] =f1_score(self._true_values[s], self.predictions[s], average='weighted')
                    if 'accuracy':
                        self.model_results[s]['accuracy'] =accuracy_score(self._true_values[s], self.predictions[s])

        else:
            ('Prediction Not Found - apply model before validating')
        pass
