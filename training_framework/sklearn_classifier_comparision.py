import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create dataframe mapper in preparation for sklearn modeling, which takes numeric numpy arrays
# https://pypi.org/project/sklearn-pandas/1.5.0/
# mapper = DataFrameMapper([
#         ('avg_dist', None),
#         ('avg_surge', None),
#         ('city', LabelBinarizer()),
#         ('phone', LabelBinarizer()),
#         ('surge_pct', None),
#         ('trips_in_first_30_days', None),
#         ('luxury_car_user', LabelEncoder()),
#         ('weekday_pct', None),
#         ('bin_avg_rating_by_driver', LabelBinarizer()),
#         ('bin_avg_rating_of_driver', LabelBinarizer()),
# ])

class Classifiers(object):
    '''
    Classifier object for fitting, storing, and comparing multiple model outputs.
    '''

    def __init__(self, classifier_list):
        self.classifiers = classifier_list
        self.classifier_names = [est.__class__.__name__ for est in self.classifiers]
        # List to store pipeline objects for classifiers
        self.pipelines = []

    def create_pipelines(self, mapper):
        for classifier in self.classifiers:
            self.pipelines.append(Pipeline([
                # ('featurize', mapper),
                ('scale', StandardScaler()),
                ('classifier', classifier)
                ]))
    
    
    def train(self, X_train, y_train):
        for pipeline in self.pipelines:
            pipeline.fit(X_train, y_train)
        
    def accuracy_scores(self, X_test, y_test):
        # Lists to store classifier test scores
        self.accuracies = []

        for pipeline in self.pipelines:            
            self.accuracies.append(pipeline.score(X_test, y_test))

        # Print results
        accuracy_df = pd.DataFrame(zip(self.classifier_names, self.accuracies))
        accuracy_df.columns = ['Classifier', 'Test Accuracies']
        print accuracy_df
            
    def plot_roc_curve(self, X_test, y_test):
    
        # Plot ROC curve for each classifier
        plt.figure(figsize=(10, 10))
        for pipeline in self.pipelines:
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            plot_roc_curve(pipeline, y_pred, y_proba, y_test)

        # 45 degree line
        x = np.linspace(0, 1.0, 20)
        plt.plot(x, x, color='grey', ls='--')
        
        # Plot labels
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity, Recall)')
        plt.title('ROC Plots')
        plt.legend(loc='lower right')
        plt.show()

    def plot_profit_curve(self, costbenefit_mat, X_test, y_test):
        
        # Plot profit curve for each classifier
        plt.figure(figsize=(10, 10))
        for pipeline in self.pipelines:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            ax, optimal_thres = plot_profit_curve(pipeline, costbenefit_mat, y_proba, y_test)
            
        # Plot labels
        # ax.legend()
        ax.set_xlabel('Prediction Threshold')
        ax.set_ylabel('Profit Per User ($)')

        # plt.xlabel('Percentage of test instances (decreasing by score)')
        # plt.ylabel('Profit')
        plt.title('Profit Curves')
        plt.legend(loc='lower left')
        plt.show()
        return optimal_thres

    def roc_curve(y_proba, y_test):
        '''
        Return the True Positive Rates, False Positive Rates and Thresholds for the
        ROC curve plot.
        
        INPUT y_proba (numpy array): predicted probabilities
        INPUT y_test (numpy array): true labels
        OUTPUT (lists): lists of true positive rates, false positive rates, thresholds 
        '''
        # thresholds = np.sort(y_proba)
        thresholds = np.linspace(0,1,num=101)
        tprs, fprs = [], []
        num_positive_cases = sum(y_test)
        num_negative_cases = len(y_test) - num_positive_cases
        for t in thresholds:
            # With this threshold, give the prediction of each instance
            predicted_positive = y_proba >= t
            # Calculate the number of correctly predicted positive cases
            true_positives = np.sum(predicted_positive * y_test)
            # Calculate the number of incorrectly predicted positive cases
            false_positives = np.sum(predicted_positive) - true_positives
            # Calculate the True Positive Rate
            tpr = true_positives / float(num_positive_cases)
            # Calculate the False Positive Rate
            fpr = false_positives / float(num_negative_cases)
            fprs.append(fpr)
            tprs.append(tpr)
        return tprs, fprs, thresholds.tolist()

    def plot_roc_curve(pipeline, y_pred, y_proba, y_test):
        '''
        Plot ROC curve with data from function above.
        '''
        tpr, fpr, thresholds = roc_curve(y_proba, y_test)

        model_name = pipeline.named_steps['classifier'].__class__.__name__
        auc = round(roc_auc_score(y_test, y_pred), 3)
        plt.plot(fpr, tpr, label='{}, AUC: {}'.format(model_name, auc))

    def standard_confusion_matrix(y_true, y_pred):
        '''
        Reformat confusion matrix output from sklearn for plotting profit curve.
        '''
        [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
        return np.array([[tn, fp], [fn, tp]])

    def plot_profit_curve(costbenefit_mat, y_proba, y_test):
        '''
        Plot profit curve.
        
        INPUTS:
        - model object
        - cost benefit matrix in the same format as the confusion matrix above
        - predicted probabilities
        - actual labels
        ''' 
        # Profit curve data
        profits = [] # one profit value for each T (threshold)
        #thresholds = sorted(y_proba, reverse=True)
        thresholds = np.linspace(0,1,num=101)
        
        # For each threshold, calculate profit - starting with largest threshold
        for t in thresholds:
            y_pred = (y_proba > t).astype(int)
            confusion_mat = standard_confusion_matrix(y_test, y_pred)
            # Calculate total profit for this threshold
            profit = sum(sum(confusion_mat * costbenefit_mat)) / len(y_test)
            profits.append(profit)
        
        # Profit curve plot
        # model_name = pipeline.named_steps['classifier'].__class__.__name__
        # max_profit = max(profits)
        m_profit_ind = np.argmax(profits)
        max_profit = profits[m_profit_ind]
        fig, ax = plt.subplots(1, figsize=(6,3))
        ax.plot(thresholds, profits, color='darkblue', label = 'Max Profit ${} Per User.'
                .format(round(max_profit,2)))
        ax.axvline(x=thresholds[m_profit_ind], color='red', linestyle='--',
                label='Optimal Threshold: {}'.format(round(thresholds[m_profit_ind], 2)))
        return ax , thresholds[m_profit_ind]