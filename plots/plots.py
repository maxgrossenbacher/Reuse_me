import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as color_map
import seaborn as sns

plt.style.use('seaborn-darkgrid') # you can custom the style here

from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_fscore_support
from matplotlib.colors import LogNorm


### ROC
def roc_curve(y_proba, y_test):
    '''
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve plot.
    
    INPUT y_proba (numpy array): predicted probabilities
    INPUT y_test (numpy array): true labels
    OUTPUT (lists): lists of true positive rates, false positive rates, thresholds 
    '''

    thresholds = np.sort(y_proba)

    tprs = []
    fprs = []

    num_positive_cases = sum(y_test)
    num_negative_cases = len(y_test) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = y_proba >= threshold
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

### Learning Curves
def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), save=False):

    plt.figure()

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("training examples")
    plt.ylabel("score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_absolute_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="cross-validation score")

    plt.legend(loc='best')
    if save:
        plt.savefig('learning_curve')
    return plt

### Confusion Matrix
def _generate_class_dicts(classes):
    """ Generate class dictionary to ints and reverse dictionary of ints to class.

        Args:
            classes (str list): List of classes
        Returns:
            class_dict (dict): classes where key = (string), values = (int)
            reverse_class_dict (dict): classes where key = (int) , values = (string)
    """
    class_dict = {}
    reverse_class_dict = {}
    counter = 0
    for i in sorted(classes):
        class_dict[i] = counter
        reverse_class_dict[counter] = i
        counter += 1
    return class_dict, reverse_class_dict

def pandas_classification_report(y_true, y_pred, target_names, save_dir):
    metrics_summary = precision_recall_fscore_support(
                    y_true=y_true,
                    y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
                    y_true=y_true,
                    y_pred=y_pred,
                    average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
                list(metrics_summary),columns=target_names,
                index=metrics_sum_index)
#             print(class_report_df)

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg

    class_report_df = class_report_df.T
    class_report_df.reset_index(inplace=True)
    class_report_df.rename(columns={'index':'class'}, inplace=True)

    if save_dir:
        class_report_df.to_csv(save_dir+'/classification_report.csv', index = False)

    return class_report_df

def plot_confusion_matrix(truth, predicted, labels={}, save_dir='',
                          title='Confusion Matrix', norm=True, suppress_values=False,
                          diagonal_values=False,
                          font_size=14,
                          cmin=0,cmax=1,
                          cut_off = 1,
                          is_recall=True,
                         model=''):
    # make confusion matrix from truth and predicted for classes
    # define the confusion matrix
    # convert to int and generate labels if string
    if isinstance(truth[0],str) and isinstance(predicted[0], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted]# if x in class_dict]

    conf_mat = confusion_matrix(truth,predicted)

    #normalise
    if norm:
        if is_recall:
            conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
        else:
            conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=0)

    fig, ax = plt.subplots(figsize=(8,8))
    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=color_map.summer, interpolation='nearest')
    cb = fig.colorbar(res)

    res.set_clim(cmin, cmax)

    # add number overlay
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if (not suppress_values or (diagonal_values and i==j)) and c>0 :
                cent = .1
                if diagonal_values:
                    cent = .3

                if norm:
                    d = round(c,2)
                    plt.text(j-cent, i+.0, d, fontsize=9)
                else:
                    plt.text(j-cent, i+.0, c, fontsize=9)

            if (i==j) and c > cut_off:
                cent= 0.3
                plt.text(j-cent, i+.0, '', fontsize=9)

    # set axes
    if labels !={}:
        _ = plt.xticks(range(len(labels)), [labels[l] for l in labels], rotation=90, fontsize=font_size)
        _ = plt.yticks(range(len(labels)), [labels[l] for l in labels],fontsize=font_size)
        report = classification_report(truth, predicted,target_names=[l for l in labels.values()])
        print(report)
        
        # class_df = pandas_classification_report(truth, predicted,target_names=[l for l in labels.values()], save_dir=save_dir)

        # # flatten classification report into a single row in a dataframe that can be exported to SQL DB for automated model performance reporting
        # cols = [l for l in class_df.columns if l != 'class']
        # col_names = {}
        # for co in cols:
        #     for c in class_df['class'].tolist():
        #         col_names[str(c)+'-'+str(co)] = class_df[class_df['class'] == c][co].tolist()[0]
        # flatten_df = pd.DataFrame(col_names,index=[0])
        # flatten_df['model'] = model
        # # save flatten_df in save directory
        # flatten_df.to_csv(save_dir+'/model_results.csv',index=False)

    plt.xlabel('Predicted',fontsize=font_size+4)
    plt.ylabel('Truth',fontsize=font_size+4)
    plt.title(title,fontsize=font_size+5)

    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('Recall', rotation=270, size=18)

    if save_dir != '':
        plt.savefig(save_dir+'/confusion_matrix.png')
    return class_df



def plot_delta_rpc(truth, predicted, predicted_2, labels={},save_name='', x_min=-0.15, x_max=0.15, y_min=-0.15, y_max=0.15):
    """
        plot recall vs precision vs count
        predicted_2 is new
    """
    # convert to int and generate labels if string
    if isinstance(truth[0],str) and isinstance(predicted[0], str) and isinstance(predicted_2[1], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]
        predicted_2 = [class_dict[x] for x in predicted_2 if x in class_dict]
    fig, ax = plt.subplots(figsize=(15,8))

    # get counts of the truth set
    y_counts_train = []
    for i in range(len(set(truth))):
        y_counts_train.append(len([x for x in truth if x==i]))

    labels = [labels[l] for l in labels]

    #plt.figure(figsize=(20,10))
    plt.title('Precision vs Recall',fontsize=15)
    plt.xlabel('Recall',fontsize=18)
    plt.ylabel('Precision',fontsize=18)


    #colors.LogNorm(vmin=Z1.min(), vmax=Z1.max())
    # plot the points
    delta_recall = recall_score(truth, predicted_2,average=None) - recall_score(truth, predicted,average=None)
    delta_precision = precision_score(truth, predicted_2,average=None) - precision_score(truth, predicted,average=None)

    print( recall_score(truth, predicted_2,average=None))
    print( recall_score(truth, predicted,average=None))
    print(delta_recall)

    
    cax = plt.scatter(
         delta_recall
        , delta_precision
        , marker = 'o'
        , s=400
        , c = recall_score(truth, predicted_2,average=None)  #_train #y_counts
        , cmap = plt.get_cmap('Spectral')
        , norm= LogNorm(np.min(recall_score(truth, predicted_2,average=None) ), vmax=np.max(recall_score(truth, predicted,average=None) ))#, cmap='PuBu_r'
        #,
        )
    plt.ylim(y_min,y_max)
    plt.xlim(x_min,x_max)
    cbar = plt.colorbar()

    #cbar.ax.set_ylabel('Original Recall', rotation=270)

    # add labels to each point
    for label, x, y in zip(labels,delta_recall, delta_precision):
        plt.annotate(
            label,
            xy = (x, y),
            xytext = (-10, 20),
            textcoords = 'offset points',
            ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.2),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('Final Recall', rotation=270, size=18)
    plt.plot((0, 0.), (0., 1), 'k-',alpha=0.5,color='g',linewidth=5)
    plt.plot((0, 1.), (0., 0), 'k-',alpha=0.5,color='g',linewidth=5)
    plt.plot((0, 0.), (0., -1), 'k-',alpha=0.5,color='r',linewidth=5)
    plt.plot((0, -1.), (0., 0), 'k-',alpha=0.5,color='r',linewidth=5)
    #plt.plot((0.6, 0.6), (0, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)

    if save_name != '':
        plt.savefig(save_name)


def plot_delta_matrix(truth, predicted, predicted_2, labels={}, save_name='',
                          title='Delta Confusion Matrix', norm=1, suppress_values=True,
                          diagonal_values=True,
                          font_size=16,
                          cmin=-0.2,cmax=0.2,
                          cut_off = 1,
                          norm_recall = True
                         ):
    # make confusion matrix from truth and predicted for classes
    # define the confusion matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np

    # convert to int and generate labels if string
    if isinstance(truth[0],str) and isinstance(predicted[0], str) and isinstance(predicted_2[1], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] if x in class_dict else class_dict['S'] for x in predicted ]
        predicted_2 = [class_dict[x] if x in class_dict else class_dict['S'] for x in predicted_2 ]

    conf_mat = confusion_matrix(truth,predicted)
    conf_mat_2 = confusion_matrix(truth, predicted_2)

    #normalise
    title_type = ''
    if norm:
        if norm_recall:
            conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
            conf_mat_2 =  conf_mat_2.astype('float')/conf_mat_2.sum(axis=1)[:, np.newaxis]
            title_type = 'Recall'
        else:
            #print('[delta_matrix]: plotting precision')
            conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=0)
            conf_mat_2 =  conf_mat_2.astype('float')/conf_mat_2.sum(axis=0)
            title_type = 'Precision'
    # take the delta map
    delta_conf_mat = (conf_mat_2 - conf_mat)

    #fig = plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(10,10))

    width = np.shape(delta_conf_mat)[1]
    height = np.shape(delta_conf_mat)[0]

    res = plt.imshow(np.array(delta_conf_mat), cmap=color_map.RdYlGn, interpolation='nearest')
    cb = fig.colorbar(res)

    res.set_clim(cmin, cmax)

    # add number overlay
    for i, row in enumerate(delta_conf_mat):
        for j, c in enumerate(row):
            if (not suppress_values or (diagonal_values and i==j)):
                cent = .1
                if diagonal_values:
                    cent = .3

                if norm:
                    d = round(c,2)
                    plt.text(j-cent, i+.0, d, fontsize=font_size-5)
                else:
                    plt.text(j-cent, i+.0, c, fontsize=font_size-5)

            if (i==j) and c > cut_off:
                cent= 0.3
                plt.text(j-cent, i+.0, 'X', fontsize=font_size-5)

    # set axes
    if labels !={}:
        _ = plt.xticks(range(len(labels)), [labels[l] for l in labels], rotation=90, fontsize=font_size)
        _ = plt.yticks(range(len(labels)), [labels[l] for l in labels],fontsize=font_size)
        print(classification_report(truth, predicted,target_names=[l for l in labels.values()]))
        print(classification_report(truth, predicted_2,target_names=[l for l in labels.values()]))

    plt.xlabel('Predicted',fontsize=font_size+4)
    plt.ylabel('Truth',fontsize=font_size+4)
    plt.title(title_type+' '+title,fontsize=font_size+5)

    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('Delta Percentage Points', rotation=270, size=18)

    if save_name != '':
        plt.savefig(save_name)

def plot_delta_matrix_old(truth, predicted, predicted_2, labels={}, save_name='',
                          title='Delta Confusion Matrix', norm=1, suppress_values=True,
                          diagonal_values=True,
                          font_size=16,
                          cmin=-0.2,cmax=0.2,
                          cut_off = 1,
                          norm_recall = True
                         ):
    # make confusion matrix from truth and predicted for classes
    # define the confusion matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np

    # convert to int and generate labels if string
    if isinstance(truth[0],str) and isinstance(predicted[0], str) and isinstance(predicted_2[1], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]
        predicted_2 = [class_dict[x] for x in predicted_2 if x in class_dict]
    #print truth, predicted, predicted_2
    conf_mat = confusion_matrix(truth,predicted)
    conf_mat_2 = confusion_matrix(truth, predicted_2)

    #normalise
    title_type = ''
    if norm:
        if norm_recall:
            conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
            conf_mat_2 =  conf_mat_2.astype('float')/conf_mat_2.sum(axis=1)[:, np.newaxis]
            title_type = 'Recall'
        else:
            #print('[delta_matrix]: plotting precision')
            conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=0)
            conf_mat_2 =  conf_mat_2.astype('float')/conf_mat_2.sum(axis=0)
            title_type = 'Precision'
    # take the delta map
    delta_conf_mat = (conf_mat_2 - conf_mat)

    #fig = plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(10,10))

    width = np.shape(delta_conf_mat)[1]
    height = np.shape(delta_conf_mat)[0]

    res = plt.imshow(np.array(delta_conf_mat), cmap=color_map.RdYlGn, interpolation='nearest')
    cb = fig.colorbar(res)

    res.set_clim(cmin, cmax)

    # add number overlay
    for i, row in enumerate(delta_conf_mat):
        for j, c in enumerate(row):
            if (not suppress_values or (diagonal_values and i==j)):
                cent = .1
                if diagonal_values:
                    cent = .3

                if norm:
                    d = round(c,2)
                    plt.text(j-cent, i+.0, d, fontsize=9)
                else:
                    plt.text(j-cent, i+.0, c, fontsize=9)

            if (i==j) and c > cut_off:
                cent= 0.3
                plt.text(j-cent, i+.0, 'X', fontsize=9)

    # set axes
    if labels !={}:
        _ = plt.xticks(range(len(labels)), [labels[l] for l in labels], rotation=90, fontsize=font_size)
        _ = plt.yticks(range(len(labels)), [labels[l] for l in labels],fontsize=font_size)
        print(classification_report(truth, predicted,target_names=[l for l in labels.values()]))
        print(classification_report(truth, predicted_2,target_names=[l for l in labels.values()]))

    plt.xlabel('Predicted',fontsize=font_size+4)
    plt.ylabel('Truth',fontsize=font_size+4)
    plt.title(title_type+' '+title,fontsize=font_size+5)

    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('Delta Percentage Points', rotation=270, size=18)

    if save_name != '':
        plt.savefig(save_name)


def plot_rpc(truth,predicted,labels={},save_name=''):
    """
        plot recall vs precision vs count
    """
    # convert to int and generate labels if string
    if isinstance(truth[0],str) and isinstance(predicted[0], str):# and isinstance(predicted_2[1], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]
        #predicted_2 = [class_dict[x] for x in predicted_2 if x in class_dict]


    # get counts of the truth set
    y_counts_train = []
    for i in range(len(set(truth))):
        y_counts_train.append(len([x for x in truth if x==i]))

    labels = [labels[l] for l in labels]

    fig, ax = plt.subplots(figsize=(14,8))
    plt.title('Precision vs Recall',fontsize=18)
    plt.xlabel('Recall',fontsize=18)
    plt.ylabel('Precision',fontsize=18)

    #colors.LogNorm(vmin=Z1.min(), vmax=Z1.max())

    # plot the points
    from matplotlib.colors import LogNorm
    cax = plt.scatter(
        recall_score(truth, predicted,average=None)
        , precision_score(truth, predicted,average=None)
        , marker = 'o'
        , s=400
        , c = y_counts_train #_train #y_counts
        , cmap = plt.get_cmap('Spectral')
        , norm= LogNorm(np.min(y_counts_train), vmax=np.max(y_counts_train))#, cmap='PuBu_r'
        #,
        )
    plt.ylim(0,1.1)
    plt.xlim(0,1.1)
    cbar = plt.colorbar()

    # add labels to each point
    for label, x, y in zip(labels, recall_score(truth, predicted,average=None), precision_score(truth, predicted,average=None)):
        plt.annotate(
            label,
            xy = (x, y),
            xytext = (-10, 20),
            textcoords = 'offset points',
            ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.2),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


    #plt.plot((0, 0.6), (0.6, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)
    #plt.plot((0.6, 0.6), (0, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)
    circle1 = plt.Circle((0, 0), 0.6, color='r',fill=False,lw=2)
    ax.add_artist(circle1)
    circle2 = plt.Circle((0, 0), 0.8, color='g',fill=False,lw=2)
    ax.add_artist(circle2)

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('Number of Class Instances', rotation=270, size=18)

    if save_name != '':
        plt.savefig(save_name)


def standard_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    '''
    Reformat confusion matrix output from sklearn for plotting profit curve.
    '''
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])

def plot_profit_curve(costbenefit_mat, y_proba, y_test, show=True):
    '''
    Plot profit curve.
    
    INPUTS:
    - model object (pipeline)
    - cost benefit matrix in the same format as the confusion matrix above
    - predicted probabilities
    - actual labels
    ''' 

    # Profit curve data
    profits = [] # one profit value for each T (threshold)
#     thresholds = sorted(y_proba, reverse=True)
    thresholds = np.linspace(0,1,num=101)
    
    # For each threshold, calculate profit - starting with largest threshold
    for T in thresholds:
        y_pred = (y_proba > T).astype(int)
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
    ax.legend()
    ax.set_title('Profit Curve')
    ax.set_xlabel('Prediction Threshold')
    ax.set_ylabel('Profit Per User ($)')
    if show:
        plt.show()
    return thresholds[m_profit_ind]

###############PLOTTING
def abline(slope, intercept):
    '''
    DESC: Plot a line from slope and intercept
    INPUT: slope(float), intercept(float)
    -----
    OUTPUT: matplotlib plot with plotted line of desired slope and intercept
    '''
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', c='b', label='Perfect Predictions')

def line_of_best_fit(x, y, ci, label,c):
    '''
    DESC: Plot a line of best fit from scatter plot
    INPUT: x-coordinates(list/array), y-coordinates(list/array), confidence-interval(float), label(str)
    -----
    OUTPUT: seaborn plot with plotted line of best fit with confidence interval and equation for line of best fit
    '''
    import seaborn as sns
    sns.regplot(x, y, fit_reg=True, scatter=True, label=label, ci=ci, color=c)
    return np.polynomial.polynomial.polyfit(x, y, 1)


def scatter_plot2d(df,col1,col2,by=False,figsize=(8,6),label=['Canola','Durum','Lentil','Hard Wheat'],vmin=0,vmax=10000,xlabel=None,
                  ylabel=None,title=None,save=False):
    '''
    DESC:
            Plot 2d histogram colored by group column
    INPUT:
            df(pd.DataFrame):           Target dataframe
            co11(str):                  First target column
            col2(str):                  Second target column
            by(str):                    group column
            label(list):                legend labels
            vmin(int):                  min value for xlim/ylim
            vmax(int):                  max value for xlim/ymin
    -----
    OUTPUT: matplotlib 2d scatter plot with perfect matching line
    '''
    if by:
        num_unique = df[by].nunique()
        unique_value = sorted(df[by].unique())
        cmap = color_map.get_cmap('hsv',num_unique+1)
        colors=[]
        for i in range(num_unique):
            colors.append(cmap(i))
        axes = []
        values = []
        k = 0
        for value,c in zip(unique_value,colors):
#            print (c,value)
            ax = plt.scatter(df.loc[df[by]==value][col1].values,df.loc[df[by]==value][col2].values,
                c=c,alpha=0.7)
            axes.append(ax)
            values.append(value)
            k += 1
        legend1 = plt.legend(tuple(axes), tuple(values),
                   scatterpoints=1, loc="best", prop={'size':8})
        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)
        #plt.xticks(np.arange(1000,6000,1000),fontsize=12)
        #plt.yticks(np.arange(1000,6000,1000),fontsize=12)
        plt.title(title,fontsize=16)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        if save:
            plt.savefig(save)
    return legend1

def actual_v_predictions_plot(actual, preds, title, fig_size=(7,7), ci=80, text=False, color='orange', label_dummies=None, save=False):
    '''
    DESC: Creates and acutal v.s. predictions plot to evaluate regressions
    INPUT: actual(list/array), preds(list/array), title(str), ci(float), pos1(tuple), pos2(tuple), save(bool)
    -----
    OUTPUT: matplotlib plot with prefect fit, line of best fit equation and plot, scatter plot of actual vs predicted values and MAPE
    '''
    from pylab import rcParams
    plt.rcParams.update({'font.size': 12, "figure.figsize":[fig_size[0],fig_size[1]]})
    plt.xlim(0,12000)
    plt.ylim(0,12000)
    best_fit_eq= line_of_best_fit(actual, preds, ci=ci, label='Line of Best Fit with {}% CI'.format(ci), c=color)
    if isinstance(label_dummies, pd.DataFrame):
        labels = label_dummies.idxmax(axis=1)
        df = pd.DataFrame({'Actual':actual.tolist(), 'Predictions':list(preds),'Labels':labels})
        legend1 = scatter_plot2d(df, 'Actual', 'Predictions', by='Labels')
    if isinstance(label_dummies, pd.Series):
        df = pd.DataFrame({'Actual':actual.tolist(), 'Predictions':list(preds),'Labels':label_dummies})
        legend1 = scatter_plot2d(df, 'Actual', 'Predictions', by='Labels')
    abline(1,0)
    # MAE, RMSE, MAPE = regression_evaluation(df['Actual'].values, df['Predictions'].values)
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.title(title)
    try:
        plt.gca().add_artist(legend1)
    except:
        pass
    if text:
        plt.text(1500,7000, text)
    if save:
        plt.savefig(save)

    plt.show()

    print('Line of Best Fit: \t\t y = {}x + {}'. format(best_fit_eq[1], best_fit_eq[0]))




############### FEATURE IMPORTANCES
def feature_importances_regressor(feature_names, model, n_features, plot=True, save=False):
    '''
    DESC: plots n most important features of XGBRegressor model.
    INPUT:  feature_names(list), model(trained model), n_features(int)
    -----
    OUTPUT: Plot top n features importances
    '''
    f_imp = model.get_booster().get_fscore()
    sorted_imp = sorted(f_imp.items(), key=lambda x: x[1], reverse=True)
    n_top_feats = sorted_imp[0:n_features]
    top_f, values = zip(*n_top_feats)
    ind =[int(f.split('f')[1]) for f in top_f]
    top_feat_names = np.array(feature_names)[ind]
    top_features = {top_feat_names[i]: (top_f[i], values[i]) for i in range(len(top_f))}
    if plot:
        plt.title("{} Most Important Features".format(n_features))
        val_sum = sum(values)
        plt.bar(range(len(n_top_feats)), (np.array(values)/val_sum) , color="g", align="center")
        plt.xlim(-1, n_features)
        plt.xticks(range(len(n_top_feats)), top_feat_names, rotation='vertical')
        plt.rcParams.update({'font.size': 13, "figure.figsize":[7,7]})
        if save:
            plt.savefig(save)
        plt.show()
    return top_features




def correlation_matrix_plot(n_top_features, df, save=False):
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    feats, split = zip(*n_top_features)
    corr = df[list(feats)].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    labels = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool))
    labels = labels.round(2)
    labels = labels.replace(np.nan,' ', regex=True)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(9,9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    mask = np.ones((len(feats), len(feats)))-mask
    ax = sns.heatmap(corr, mask=mask, cmap=ListedColormap(['white']), annot=labels,cbar=False, fmt='', linewidths=1.5)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()




def error_by_crop(train, actual, model, crop):
    '''
    DESC: Creates an actaul v.s predictions plot based on particular crop_type.
    NOTE: Make sure to reset the X_test index before running this functio
    INPUT: X_test(array), y_test(array), crop_type(str), model(sklearn-model), pos1(tuple), pos2(tuple)
    -----
    OUTPUT: Actual v.s. prediction plot for particular crop_type.
    '''
    t = train.reset_index()
    crop_test = t[t['crop_2017']==crop]
    crop_ind = list(crop_test.index.values)
    crop_actual = actual.iloc[crop_ind]
    crop_test.drop('index', axis=1, inplace=True)
    crop_predictions = model.predict(crop_test.values)
    print('Number of Fields with', str(crop) + ' = ', len(crop_predictions))
    if crop_test.shape[0] > 0:
        metric_val = regression_evaluation(crop_actual, crop_predictions)
        print(10*'-')
        return crop_actual, crop_predictions, metric_val




def predicted_actual_dist(actual, predictions, bins, save=False):
    plt.figure(figsize=(7,7))
    bins = np.linspace(0, 9000, bins)
    plt.hist(predictions, bins=bins, alpha=0.5, label='Predicted Yield')
    plt.hist(actual, bins=bins, color='g', alpha =0.5, label='Actual Yield')
    plt.xlabel('Yield', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    plt.legend(loc='best')
    plt.title('Distribution of Yield Predictions', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()



def histogram2d(actual, predictions, save=False):
    plt.hist2d(actual, predictions)
    plt.colorbar()
    plt.title('2D Histogram Actual v.s. Predicted Yield', fontsize=16)
    plt.xlabel('Predicted Yield', fontsize=16)
    plt.ylabel('Actual Yield', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def scatter_plot2d_colored(df,col1,col2,by=False,figsize=(8,6),label=['Canola','Durum','Lentil','Hard Wheat'],\
                            xlabel=None, line=False, ylabel=None,title=None, vmin=0, vmax=10000, save=False, text=None):
    '''
    DESC:
            Plot 2d histogram colored by group column
    INPUT:
            df(pd.DataFrame):           Target dataframe
            co11(str):                  First target column
            col2(str):                  Second target column
            by(str):                    group column
            label(list):                legend labels
            vmin(int):                  min value for xlim/ylim
            vmax(int):                  max value for xlim/ymin

    -----
    OUTPUT: matplotlib 2d scatter plot with perfect matching line
    '''
    if by:
        num_unique = df[by].nunique()
        unique_value = sorted(df[by].unique())
        cmap = color_map.get_cmap('hsv',num_unique+1)

        colors=[]
        for i in range(num_unique):
            colors.append(cmap(i))

        for value,c in zip(unique_value,colors):
            print (c, value)
            plt.scatter(df.loc[df[by]==value][col1].values,df.loc[df[by]==value][col2].values,
                c=c,alpha=0.7)

        plt.legend(label, loc='best')
        if line:
            abline(1,0)
        if text:
            plt.text(text[0], text[1], text[2])

        # plt.xlim(vmin, vmax)
        # plt.ylim(vmin, vmax)

        #plt.xticks(np.arange(1000,6000,1000),fontsize=12)
        #plt.yticks(np.arange(1000,6000,1000),fontsize=12)

        plt.title(title,fontsize=16)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(save)

        plt.show()

# Train test split visualization by particular colname
def train_test_split_vis(train, test, colname, title, save=False , fontsize=15):
    '''
    DESC: Function to visualize Train test split by colname \ works best with categorical features
    INPUT: Train(DataFrame), Test(DataFrame), colname(str), title(str), fontsize(int)
    -----
    OUTPUT: Figure
    '''
    train = train[colname].value_counts(normalize=True).reset_index()
    test = test[colname].value_counts(normalize=True).reset_index()

    merged = train.join(test, lsuffix='_train', rsuffix='_test')
    merged.set_index('index_train', inplace=True)
    merged.drop('index_test', axis=1, inplace=True)

    ax = merged.plot.barh(figsize=(10,5), title=title)
    bigger_fonts(ax, fsize=fontsize)
    plt.xlabel('% Counts')
    plt.ylabel('Crop ID')
    plt.tight_layout()
    if save:
        plt.savefig(title)
    plt.show()


# function to create bigger fonts
def bigger_fonts(ax, fsize):
    '''
    DESC: Function to create bigger font sizes on matplotlib plots
    INPUT: matplotlib ax, fsize(int)
    -----
    OUTPUT: Figure with font size specified by fsize
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
