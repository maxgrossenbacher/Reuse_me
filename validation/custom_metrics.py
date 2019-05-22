import numpy as np
import pandas as pd

def fuzzy_accuracy(actual, predictions, std):
    fuzzy_accuracy = pd.concat([actual, std], axis=1)
    fuzzy_accuracy = pd.concat([fuzzy_accuracy.reset_index(), pd.Series(predictions)], axis=1)
    fuzzy_accuracy.drop('index', axis=1, inplace=True)
    fuzzy_accuracy.rename(columns = {fuzzy_accuracy.columns[0]: "Yield",
                                    fuzzy_accuracy.columns[1]: "Yield std",
                                    fuzzy_accuracy.columns[2]: "Predicted Yield"}, inplace=True)


    fuzzy_accuracy['lower_bound'] = fuzzy_accuracy['Yield'] - fuzzy_accuracy['Yield std']
    fuzzy_accuracy['upper_bound'] = fuzzy_accuracy['Yield'] + fuzzy_accuracy['Yield std']

    fuzzy_accuracy['within_std'] = (fuzzy_accuracy['Predicted Yield'] > fuzzy_accuracy['lower_bound']) & \
                            (fuzzy_accuracy['Predicted Yield'] < fuzzy_accuracy['upper_bound'])
    fuzzy_accuracy['within_2std'] = (fuzzy_accuracy['Predicted Yield'] > fuzzy_accuracy['lower_bound']*2) & \
                            (fuzzy_accuracy['Predicted Yield'] < fuzzy_accuracy['upper_bound']*2)

    def which_std(row):
        if row['within_std'] == True:
            return 0
        elif row['within_2std'] == True:
            return 1
        else:
            return 2
    fuzzy_accuracy['which_std'] = fuzzy_accuracy.apply(which_std, axis=1)

    accuracy = (fuzzy_accuracy['within_std'] == True).sum() / fuzzy_accuracy.shape[0]
    print('Accuracy Score: {}%'.format(accuracy*100))
    return accuracy, fuzzy_accuracy

def mean_absolute_precentage_error(actual, preds):
    '''
    DESC: Function to calculate Mean Absolute Precentage Error (MAPE)
    INPUT: actual(np.array), preds(np.array)
    -----
    OUTPUT: MAPE
    '''
    import numpy as np
    y_true, y_pred = np.array(actual), np.array(preds)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

    # functions for evaluation metrics (mean absolute precision)
def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
