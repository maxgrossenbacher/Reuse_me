import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid') # you can custom the style here


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