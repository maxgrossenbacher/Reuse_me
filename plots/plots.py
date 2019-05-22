import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid') # you can custom the style here

from sklearn.learning_curve import learning_curve


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

def dropping(df, drop_cols):
    if drop_cols:
        for col in drop_cols:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
        for c in df.columns:
            if df[c].dtype not in [float, int, bool]:
                df[c]=pd.to_numeric(df[c], errors='coerce')
        return df

def bagging_histogram(truths, preds,cut_off=-1):
    acc = []
    pre = []
    rec = []
    f1s = []
    import sklearn as sk

    truth = truths
    for i in range(len(preds)):
        if cut_off < 1 and cut_off > 0:
            pred = preds[i][:,1]
            prediction =  pred > cut_off
        else:
            prediction = preds[i]
        acc.append(sk.metrics.accuracy_score(prediction,truth[i]))
        pre.append(sk.metrics.precision_score(prediction,truth[i]))
        rec.append(sk.metrics.recall_score(prediction,truth[i]))
        f1s.append(sk.metrics.f1_score(prediction,truth[i]))

    plt.figure(figsize=(8,5))
    plt.hist(acc, bins = [0. + x*0.025 for x in range(40)], alpha=0.6,label='Accuracy',color='b')
    plt.axvline(linewidth=4, color='r',x=np.mean(acc), label='Mean(Acc) = '+str(round(np.mean(acc)*1000)/10),c='b')


    plt.hist(pre, bins = [0. + x*0.025 for x in range(40)], alpha=0.6,label='Precision',color='y')
    plt.axvline(linewidth=4,x=np.mean(pre), label='Mean(Pre) = '+str(round(np.mean(pre)*1000)/10),c='y')

    plt.hist(rec, bins = [0. + x*0.025 for x in range(40)], alpha=0.6,label='Recall',color='g')
    plt.axvline(linewidth=4,x=np.mean(rec), label='Mean(Rec) = '+str(round(np.mean(rec)*1000)/10),c='g')

    plt.hist(f1s, bins = [0. + x*0.025 for x in range(40)], alpha=0.6,label='F1',color='r')
    plt.axvline(linewidth=4,x=np.mean(f1s), label='Mean(F1) = '+str(round(np.mean(f1s)*1000)/10),c='r')


    plt.xlabel('Metric',size=18)
    plt.ylabel('Count',size=18)
    plt.legend(fontsize=13, loc='upper left')
    plt.title('Full Features',size=20)


def feat_importance_plot(gbm, feature_list, number=15,flag = True):
    '''
    Plot all feature importance and top importance feature in hist
    Args:
        number(int): how many top importance
        folder_name: model folder name
        flag: True for xgboost and False for random forest or other models
    Returns:
        feature importance
    '''
    import dill
    import xgboost as xgb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np


    #features = dill.load(open(folder_name+'/features.p','rb'))


    bst = gbm.booster()
    importance = bst.get_fscore()

    newdict  = dict((feature_list[int(key.replace('f',''))],value) for (key, value) in importance.items())
    df_feat = pd.DataFrame({'Importance': list(newdict.values()), 'Feature': list(newdict.keys())})

    df_feat = df_feat.sort_values(by = 'Importance', ascending=[False])

    fig = plt.figure(1, [6, 5])

    temp = df_feat[0:number]
    names = temp.Feature
    x = np.arange(len(names))
    plt.bar(x, temp.Importance.ravel())
    plt.xlabel('Feature',size=16)
    plt.ylabel('Normalised Importance',size=16)
    plt.title('Feature Importance Ranking',size=16)
    _ = plt.xticks(x + 0.5, names, rotation=90)
    plt.ylim([0, np.max(temp.Importance.ravel())+1.4])
    plt.xlim([-1, len(names)+1])
    plt.show()

    x = list(range(0,len(df_feat['Importance'])))
    y = df_feat.Importance.ravel()

    plt.plot(x,y, linestyle='-', linewidth=1.0,)
    plt.xlabel('Number of Features')
    plt.ylabel('Normalized Feature Importance')
    plt.title('Feature Importance of all Features')
    plt.ylim([0, np.max(y)+1.4])
    plt.show()

    df_feat.to_csv('xgb_fcore_importance.csv',index=False)


def run_xgbags(df, features, params, iterations=20, class_col='y'):
    import validation as val
    import xgboost as xgb
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    preds = []
    truths = []
    for i in range(iterations):
        print('[model]:',i)
        if i == 0:
            # just so that we have the first seed as the one above
            df = val.add_train_validation_split(df,validation_size=0.05, test_size=0.2, random_seed=42)
        else:
            df = val.add_train_validation_split(df,validation_size=0.05, test_size=0.2, random_seed=(i*2)+41)
        X_train = df[df.split=='Train']
        X_val  = df[df.split=='Validation']
        X_test  = df[df.split=='Test']

        eval_set = [(X_train[features].values,X_train[class_col]),
                        (X_val[features].values,X_val[class_col])]

        xgb_model = xgb.XGBClassifier(**params)

        gbm = xgb_model.fit(X_train[features].values,
                X_train[class_col].values,
                eval_metric="logloss",
                            eval_set=eval_set,
                            verbose=100,
                            early_stopping_rounds=50)



        pred = gbm.predict_proba(X_test[features].values)
        preds.append(pred)
        truths.append(X_test[class_col].values)

        pred_int = gbm.predict(X_test[features].values)

        print('   [scores]: argmax accuracy -> ',accuracy_score(X_test[class_col].values, pred_int))
        print('   [scores]: argmax f1score -> ',f1_score(X_test[class_col].values, pred_int))

    return truths, preds

def plot_accuracy_cut(df, truth_col='Tag', prob_col_prefix='Probability ', default_class='S', show_plot=True, save_dir=None, x_min=-0.01, x_max=1.05, y_min=-0.05, y_max=1.05):
    #
    # this makes an accuracy curve for every class in a multiclass problem
    # - does so by comparing 1 vs ALL
    # - this one is a mess, need to refactor if time
    #
    import copy
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, auc

    def _compute_accuracy(cut,yhat,y,stat='f1_score'):
        #print cut
        yhat[yhat<cut]  = 0
        yhat[yhat>=cut] = 1

        #w1 = np.where(y==1)
        #w0 = np.where(y==0)

        #c_rate = np.mean( y==yhat )
        if stat=='f1_score':
            f1_s = f1_score(y,yhat)
        elif stat == 'precision_score':
            f1_s = precision_score(y,yhat)
        elif stat == 'recall_score':
            f1_s = recall_score(y,yhat)


        return f1_s

    from scipy import interp
    from itertools import cycle
    import numpy as np
    colormap = plt.cm.nipy_spectral #gist_ncar #nipy_spectral, Set1,Paired

    lw = 1.5
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    #
    truth = df[truth_col].values
    class_dict, _ = _generate_class_dicts(set(truth))
    colors = [colormap(i) for i in np.linspace(0, 0.9,len(class_dict.keys()))]

    class_names = list(class_dict.keys())
    class_names.sort()

    if show_plot:
        stats = ['f1_score','precision_score','recall_score', 'roc']
    else:
        stats = ['f1_score']

    max_f1_dict = {}
    for stat in stats:
        for k, color in zip(class_names, colors):
            if stat != 'roc':
                cut_off = -1

                # don't do anything for default class
                if k == default_class:
                    y = [1 if x == k else 0 for x in truth]
                    size_default = len(np.where(np.array(y)==1)[0])
                    size_total = len(y)
                    continue

                # multi class either 1 or 0
                y = [1 if x == k else 0 for x in truth]
                scores = df[prob_col_prefix+k].values

                # run over the bins to compute the statistics
                bins = [x*0.01 for x in range(101)]
                f1 = []
                max_val = -1
                for i in bins:
                    labels = copy.deepcopy(np.array(y))
                    preds  = copy.deepcopy(np.array(scores))

                    current_val = _compute_accuracy(i,preds,labels,stat)
                    f1.append(current_val)
                    if current_val > max_val:
                        max_val = current_val
                        cut_off = i

                # sace the cut_off if we're using f1_score
                if stat == 'f1_score':
                    max_f1_dict[k] = cut_off
            if stat == 'roc':
                fpr, tpr, threshold = roc_curve(y, scores)
                roc_auc = auc(fpr, tpr) 

            if show_plot:
                if stat != 'roc':
                	plt.plot(bins,f1,label=k,color=color)
                else:
                    plt.plot(fpr,tpr,label=k,color=color)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')

            if show_plot:
                if stat != 'roc':
                	plt.plot(bins,f1,label=k,color=color)
                else:
                    plt.plot(f1[0],f1[1],label=k,color=color)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')

        ##
        if show_plot:
            fig = plt.figure(1, figsize=(7, 7))
            plt.gca().xaxis.grid(True)
            #plt.xticks(list(plt.xticks()[0]) + [0.45])
            plt.xlabel('Probability Cutoff', fontsize =16)
            plt.ylabel('Score', fontsize =16)


            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            if stat == 'roc':
            	plt.xlabel('False Positive Rate')
            	plt.ylabel('True Positive Rate')
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            plt.title(stat.upper().replace('_',' ')+' [1 vs ALL]',fontsize=18)
            #plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5), title='CLASSES [S = %d/%d]'%(size_default,size_total))
            plt.legend(loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5), title='CLASSES')

            plt.show()
            if save_dir:
                plt.savefig(save_dir+'/'+stat.upper().replace('_',' ')+' [1 vs ALL]')

    return max_f1_dict

def plot_logit(truth, predicted_probability, default_class = 'S',title_str=''):
    import copy
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score
    #from sklearn.metrics import f1_score

    # convert to int and generate labels if string
    if isinstance(truth[0],str):
        #class_dict, labels = _generate_class_dicts(set(truth))
        truth = [0 if x == default_class else 1 for x in truth]

    #sens = []
    #spec = []

    recall_s = []
    precision_s = []

    clas = []
    f1 = []

    def _make_logit(cut,mod,y):
        #print cut
        yhat = mod
        yhat[yhat<cut]  = 0
        yhat[yhat>=cut] = 1

        w1 = np.where(y==1)
        w0 = np.where(y==0)
        #print yhat
        #print w1
        #print w0

        #sensitivity = np.mean( yhat[w1] == 1 )
        #specificity = np.mean( yhat[w0] == 0 )

        c_rate = np.mean( y==yhat )
        f1_s = f1_score(y,yhat)

        r_s = recall_score(y,yhat)
        p_s = precision_score(y,yhat)
        #print sensitivity,specificity,c_rate,f1_s
        #sens.append(sensitivity)
        #spec.append(specificity)

        recall_s.append(r_s)
        precision_s.append(p_s)

        clas.append(c_rate)
        f1.append(f1_s)

    bins = [x*0.02 for x in range(50)]
    truth = np.array(truth)
    predicted_probability = np.array(predicted_probability)
    for i in bins:
        labels = copy.deepcopy(truth)
        preds = copy.deepcopy(predicted_probability)
        _make_logit(i,preds,labels)

    ## Sensitivity: true positive rate -> propotion of positives that are correctly identified as such (percentage of sick people who are correctly identified as sick)
    ## Specificity: true negative rate -> propotion of negatives that are correctly identified as such (percentage of healthy people who are correctly identified as healthy)
    ## A perfect classifier would have 100% sensitivity and specificity
    fig = plt.figure(1, figsize=(5, 5))
    plt.plot(bins,recall_s,label="Recall")
    plt.plot(bins,precision_s,label="Precision")

    #plt.plot(bins,clas,label="Classification Rate")

    plt.plot(bins,f1,label="F1")
    plt.gca().xaxis.grid(True)
    #plt.xticks(list(plt.xticks()[0]) + [0.45])
    plt.xlabel('Logit Cutoff', fontsize =20)
    plt.ylabel('Value', fontsize =20)

    plt.title(title_str,fontsize=20)
    plt.legend()
    plt.show()


def plot_roc(df, truth_col='Tag', prob_col_prefix = 'Probability ', default_class = 'S', x_min=-0.01, x_max=0.4, y_min=-0.05, y_max=1.05):
    #
    # this makes an ROC curve for every class in a multiclass problem
    # - does so by comparing 1 vs ALL
    #
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from itertools import cycle
    import numpy as np
    colormap = plt.cm.nipy_spectral #gist_ncar #nipy_spectral, Set1,Paired


    lw = 1.5
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    #
    truth = df[truth_col].values
    class_dict, _ = _generate_class_dicts(set(truth))
    colors = [colormap(i) for i in np.linspace(0, 0.9,len(class_dict.keys()))]

    class_names = list(class_dict.keys())
    class_names.sort()
    for k, color in zip(class_names, colors):
        if k == default_class:
            y = [1 if x == k else 0 for x in truth]
            size_default = len(np.where(np.array(y)==1)[0])
            size_total = len(y)
            continue

        y = [1 if x == k else 0 for x in truth]
        #scores = [p if x == k else (1-p) for x,p in zip(predicted_class, probability)]
        scores = df['Probability '+k].values

        fpr, tpr, thresholds = roc_curve(y, scores)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='%s (%d) %0.3f' % (k, len(np.where(np.array(y)==1)[0]),roc_auc))

    mean_tpr /= (len(class_dict.keys())-1)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    fig = plt.figure(1, figsize=(5, 5))

    plt.plot(mean_fpr, mean_tpr, color='k', linestyle='--',
         label='MEAN ROC %0.3f'%(mean_auc), lw=4.5)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='b')

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC [1 vs ALL]')
    plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5), title='CLASSES [S = %d/%d]'%(size_default,size_total))
    plt.show()


def plot_statistic_trend(df, original, column_list, label_list = None, truth_col='Tag', stat='f1_score', tag_rev_class_dict={}):
    #
    # plot a trend of the different columns for accuracy comparisons between models
    #
    from sklearn.metrics import f1_score, recall_score, precision_score
    import numpy as np
    import matplotlib.pyplot as plt
    import copy

    colormap = plt.cm.nipy_spectral

    # get the labels of the class
    labels = df[truth_col].unique()
    labels.sort()
    colors = [colormap(i) for i in np.linspace(0, 0.9,len(labels))]


    if stat == 'f1_score':
        st_orig = f1_score(df[truth_col], df[original], average=None,labels=labels)
    elif stat == 'recall_score':
        st_orig = recall_score(df[truth_col], df[original], average=None, labels=labels)
    elif stat == 'precision_score':
        st_orig = precision_score(df[truth_col], df[original], average=None, labels=labels)
    else:
        raise ValueError()

    #
    st_orig

    # compute stats for each col
    # for each class
    statistics = []
    statistics.append(st_orig)

    for c in column_list:
        if stat == 'f1_score':
            st = f1_score(df[truth_col], df[c], average=None,labels=labels)
        elif stat == 'recall_score':
            st = recall_score(df[truth_col], df[c], average=None, labels=labels)
        elif stat == 'precision_score':
            st = precision_score(df[truth_col], df[c], average=None, labels=labels)
        else:
            raise ValueError()
        statistics.append(st)


    # transpose such that each set
    statistics = [list(i) for i in zip(*statistics)]


    #fig,ax = plt.figure(1, figsize=(5, 5))
    fig, ax = plt.subplots(1,figsize=(5,5))
    counter = 0
    # make the plot
    for s,color in zip(statistics,colors):
        plt.plot(s,color=color,lw=3,label=labels[counter])
        counter+=1

    #plt.xlabel('Model Varieties',size=15)
    plt.ylabel(stat,size=15)
    plt.title('Model Varieties',size=15)

    #plt.legend()
    if tag_rev_class_dict:
        # fix the legend manually
        import matplotlib.lines as mlines
        lines = []
        for i in range(len(labels)):
             lines.append( mlines.Line2D([], [], color=colors[i],
                                  markersize=15, label=tag_rev_class_dict[labels[i]]))

        plt.legend(loc="center left", fontsize=12, bbox_to_anchor=(1, 0.5), title='CLASSES', handles=lines)
    else:
        plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5), title='CLASSES')

    #ax.set_xticklabels(label_list)
    plt.xticks(range(len(label_list)), label_list, rotation='vertical',size=13)



def plot_classes(x, y):
    """
        plot2d histogram where x and y are str lists
        Args:
            x (str list):  some list of string categories
            y (str list): some other list of string categories
        Returns:
            plt (image to screen): of 2D histogram
    """

    from itertools import count
    import numpy as np
    classes = sorted(set(x))
    class_dict = dict(zip(classes, count()))
    class_map = lambda x: class_dict[x]

    classes_y = sorted(set(y))
    class_dict_y = dict(zip(classes_y, count()))
    class_map_y = lambda x: class_dict_y[x]

    plt.figure(figsize=(12,8))

    fig, ax = plt.subplots()
    _x = [i for i in map(class_map, x)]
    _y = [j for j in map(class_map_y, y)]

    im = ax.hexbin(_x, _y, gridsize=20,bins='log', cmap=plt.cm.YlOrRd_r)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('log10(N)')

    plt.xticks(np.arange(len(classes)), classes, rotation=90)
    plt.yticks(np.arange(len(classes_y)), classes_y)

    plt.xlabel('PoS',fontsize=15)
    plt.ylabel('Tag',fontsize=15)

    plt.ylim(-1,len(classes_y))
    plt.xlim(-1,len(classes))


def plot_cm_by_row(predicted,truth,is_recall=True):
    """
        Make bar chart of a CM matrix for more accurate visualsiation of classes
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np

    conf_mat = confusion_matrix(truth,predicted)
    conf_mat =  conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]

    rc = 0
    labels = tag_rev_class_dict
    for row in conf_mat:
        plt.figure(figsize=(6,3))
        plt.bar(range(0,len(row)),row)
        plt.title(labels[rc])
        plt.plot((0, len(tag_rev_class_dict)), (0.6, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)

        plt.ylabel('Recall')
        plt.ylim(0,1)
        _ = plt.xticks([x+0.5 for x in range(len(tag_rev_class_dict))], [tag_rev_class_dict[l] for l in tag_rev_class_dict], rotation=90, fontsize=8)
        rc+=1

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
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np



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

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
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

        # create classification report and save as csv
        import pandas as pd
        from sklearn.metrics import precision_recall_fscore_support
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

        class_df = pandas_classification_report(truth, predicted,target_names=[l for l in labels.values()], save_dir=save_dir)

        # flatten classification report into a single row in a dataframe that can be exported to SQL DB for automated model performance reporting
        cols = [l for l in class_df.columns if l != 'class']
        col_names = {}
        for co in cols:
            for c in class_df['class'].tolist():
                col_names[str(c)+'-'+str(co)] = class_df[class_df['class'] == c][co].tolist()[0]
        flatten_df = pd.DataFrame(col_names,index=[0])
        flatten_df['model'] = model
        # save flatten_df in save directory
        flatten_df.to_csv(save_dir+'/model_results.csv',index=False)

    plt.xlabel('Predicted',fontsize=font_size+4)
    plt.ylabel('Truth',fontsize=font_size+4)
    plt.title(title,fontsize=font_size+5)

    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('Recall', rotation=270, size=18)

    if save_dir != '':
        plt.savefig(save_dir+'/confusion_matrix.png')
    return class_df, flatten_df



def plot_delta_rpc(truth, predicted, predicted_2, labels={},save_name='', x_min=-0.15, x_max=0.15, y_min=-0.15, y_max=0.15):
    """
        plot recall vs precision vs count
        predicted_2 is new
    """
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    # convert to int and generate labels if string
    if isinstance(truth[0],str) and isinstance(predicted[0], str) and isinstance(predicted_2[1], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]
        predicted_2 = [class_dict[x] for x in predicted_2 if x in class_dict]

    import numpy as np
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

    from matplotlib.colors import LogNorm
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

    res = plt.imshow(np.array(delta_conf_mat), cmap=plt.cm.RdYlGn, interpolation='nearest')
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

    res = plt.imshow(np.array(delta_conf_mat), cmap=plt.cm.RdYlGn, interpolation='nearest')
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
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    import numpy as np

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

def feat_importance_plot(number,folder_name='',flag = True):
    '''
    Plot all feature importance and top importance feature in hist
    Args:
        number(int): how many top importance
        folder_name: model folder name
        flag: True for xgboost and False for random forest or other models
    Returns:
        feature importance
    '''
    import dill
    import xgboost as xgb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np


    features = dill.load(open(folder_name+'/features.p','rb'))

    if flag == True:
        bst = xgb.Booster({'nthread':4})
        bst.load_model(folder_name+'/xg.model')
        importance = bst.get_fscore()

        newdict  = dict((features[int(key.replace('f',''))],value) for (key, value) in importance.items())
        df_feat = pd.DataFrame({'Importance': list(newdict.values()), 'Feature': list(newdict.keys())})

    else:
        import os
        dire = os.getcwd()

        from sklearn.externals import joblib
        filename = os.path.join(dire, folder_name+'/model.pkl')
        clf = joblib.load(filename)

        df_feat = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})

    df_feat = df_feat.sort_values(by = 'Importance', ascending=[False])

    fig = plt.figure(1, [20, 8])

    temp = df_feat[0:number]
    names = temp.Feature
    x = np.arange(len(names))
    plt.bar(x, temp.Importance.ravel())
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    _ = plt.xticks(x + 0.8, names, rotation=98)

    plt.show()

    x = list(range(0,len(df_feat['Importance'])))
    y = df_feat.Importance.ravel()

    plt.plot(x,y, linestyle='-', linewidth=1.0,)
    plt.xlabel('Number of Features')
    plt.ylabel('Normalized Feature Importance')
    plt.title('Feature Importance of all Features')

    plt.show()

    df_feat.to_csv('xgb_fcore_importance.csv',index=False)

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
        cmap = plt.cm.get_cmap('hsv',num_unique+1)
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
        cmap = plt.cm.get_cmap('hsv',num_unique+1)

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

def cumulative_histogram(y_act, y_pred, n_bins_pred=25, max_pred=10000, max_delta=1000,  n_bins_delta=100):
    """
        create cumulative residual histogram normalised to counts
    """
    # set up bin widths
    x_bin_width = (max_pred/n_bins_pred)
    y_bin_width = (max_delta/n_bins_delta)
    x_bins = [int((x)*x_bin_width) for x in range(n_bins_pred+1)]
    y_bins = [int((x-np.floor(n_bins_delta))*y_bin_width) for x in range(n_bins_delta*2+1)]

    # sum all
    cumulative_hist = np.zeros((len(x_bins)-1,len(y_bins)-1))
    for pred, act in zip(y_pred, y_act):
        # get the counts using the plot
        bins, x, y, _ = plt.hist2d(
                          pred,
                          (pred - act),
                          bins = (x_bins, y_bins))

        # normalise to the total number
        cumulative_hist += bins/np.sum(bins)

    # suppress plot output
    plt.close()
    return cumulative_hist, x, y


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'same')
    return sma

def compute_confidence_bands(cumulative_hist, x, y, conf_levels=[0.05, 0.95], n_bins_pred=25, max_pred=10000, show_plot=True, smooth = True):
    """
        compute confidence band list, returns as delta to actual per bin
    """
    import matplotlib.cm as cm

    # normalise to each x-axis bin
    norm_hist = cumulative_hist / np.sum(cumulative_hist, axis=1, keepdims=True)

    # cumulative sum
    cumulative_hist_sum = np.cumsum(norm_hist, axis=1)

    # loop over different quantiles and read off where y is
    conf_bands = {}
    for conf_level in conf_levels:
        x_bins = []
        conf_band = []
        # loop over the x values
        fix_first = False
        for ix, x_bin in enumerate(cumulative_hist_sum):
            # find where y percentile > conf_level and save it for each step in x
            is_exceed = False
            for iy, x_val in enumerate(x_bin):
                if x_val >= conf_level:
                    conf_band.append(y[iy])
                    is_exceed = True
                    break

            # defaults to previous value if threshold not found
            if not is_exceed:
                if len(conf_band) > 0:
                    print('[warning]: no data in x-bin, defaulting to previous value...')
                    conf_band.append(conf_band[-1])
                else:
                    fix_first = True
                    conf_band.append(0)
            x_bins.append(x[ix] + (max_pred/n_bins_pred)/2)

        # fix the first band if zero
        if fix_first:
            last_val = -1
            for ic, c in reversed(list(enumerate(conf_band))):
                if c == 0:
                    conf_band[ic] = last_val
                else:
                    last_val = c

        # save for output
        conf_bands[conf_level] = conf_band
        # print(conf_band)

        if smooth:
            conf_bands[conf_level] = moving_average(conf_band, 7)
            # print(conf_band)

    if show_plot:
        fig, ax = plt.subplots()
        cs = ax.imshow(norm_hist.T, origin='bottom', extent=(np.min(x),np.max(x),np.min(y),np.max(y)),
                  aspect='auto', cmap=cm.Reds)
        cb = fig.colorbar(cs)
        cb.set_label('Predicted Bucket Density',size=15,rotation=270, labelpad=15)
        ax.set_xlabel('Predicted',size=14)
        ax.set_ylabel('Predicted - Actual',size=14)
        for conf_band in conf_bands:
            # note we add on just for the plot to extend to end
            # print(([x[0]] + x_bins + [x[-1]]),([conf_bands[conf_band][0]] + conf_bands[conf_band]+[conf_bands[conf_band][-1]]) )
            plt.plot(([x[0]] + x_bins + [x[-1]]), ([conf_bands[conf_band][0]] + conf_bands[conf_band]+[conf_bands[conf_band][-1]]),  lw=4, label='{0:.2f}'.format(round(conf_band,2)))

        plt.legend(loc='upper right',fontsize=7,title='C.L.')


    return conf_bands, x, x_bins

def convert_to_cl(values, conf_list, x):
    """
        create conversion of values
    """
    # assert len(conf_list) == len(x) - 1

    corrected_val = []
    for c in values:
        # no value... exceeds
        if c <= x[0]:
            corrected_val.append(conf_list[0])
        elif c > x[-1]:
            corrected_val.append(conf_list[-1])
        else:
            for ix, x_lim in enumerate(x[:-1]):
                if c > x[ix] and c <= x[ix+1]:
                    corrected_val.append(conf_list[ix])
                    break

    return corrected_val

def sort_x_y(X,Y):
    """
    sort X using Y, helper function
    """
    return [x for _,x in sorted(zip(Y,X))]