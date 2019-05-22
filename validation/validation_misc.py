import numpy as np
import pandas as pd

def get_top_n_class_predictions(model, preds_probability, n=5):
    results_df = pd.DataFrame(
        columns=model.classes_, data=preds_probability)
    sorted_probabilities = np.flip(results_df.values.argsort(), 1)
    return sorted_probabilities[:,:n]