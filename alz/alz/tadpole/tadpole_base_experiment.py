"""This is an example base experiment combining multiple modalities and running an 
experiment to predict a diagnosis or cognitive score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import alz.tadpole.tadpole_processing_helpers as helpers
except ImportError:
    import tadpole_processing_helpers as helpers
import pickle
import sklearn
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

pickle_path = "out/merged_data.pkl"

with open(pickle_path, "rb") as pickle_file:
    merged_dataset = pickle.load(pickle_file)


def get_data_for_experiment_with_modalities(modalities, target_modality, target_column, data, columns_to_drop=None, verbose=False):
    """Clean and return the data with the particular modalities and desired target
    
    Args:
        modalities (list): The list of modalities to use as features
        target_modality (str): The name of the modality with the desired target
        target_column (str): The name of the column within the target
        data (dict): The saved dataset
        columns_to_ignore (list, optional): Any columns to drop from the dataframe before cleaning the data. Defaults to None.
        verbose (bool, optional): Whether to display verbose print messages. Defaults to False.
    
    Returns:
        tuple: The features and targets cleaned up for experimentation
    """

    df = pd.concat([pd.DataFrame.from_records(data[modality]).T for modality in modalities+[target_modality]], axis=1)

    if columns_to_drop is not None:
        df = df.drop(columns_to_drop, axis=1)

    if verbose:
        print(df.shape)
        print("Row analysis of NaNs")
        print(df.isna().sum(axis=0))

    df = df.dropna(axis=0, how="all")
    if verbose:
        print(df.isna().sum(axis=0))
        print(df.shape)
        print("Column analysis of NaNs")
        print(df.isna().sum(axis=1))

    df = df.dropna(axis=1, how="all")
    if verbose:
        print(df.isna().sum(axis=1))
        print(df.shape)

    df = df.dropna(axis=0, how="any")
    if verbose:
        print(df.isna().sum().sum())
        print(df.shape)

    targets = df[[target_column]]
    if target_column == "DX":
        targets = targets["DX"].apply(convert_string_to_integer_diagnosis)
    

    features = df.drop(target_column, axis=1)

    return features, targets


# # Ignore Ecog Columns
# ecog_columns = [column for column in df.columns if column.startswith("Ecog")]
# df = df.drop(ecog_columns, axis=1)

# # Ignore RAVLT columns
# ravlt_columns = [column for column in df.columns if column.startswith("RAVLT")]
# df = df.drop(ravlt_columns, axis=1)

# Select rows that don't have NA for certain cognitive tests
# important_columns = ["DX", "ADAS11", "MMSE"]
# df = df.dropna(subset=important_columns)




def convert_string_to_integer_diagnosis(string_diagnosis):
    """Convert string diagnosis to integer for machine learning prediction
    
    Args:
        string_diagnosis (str): The string diagnosis
    
    Returns:
        int: An integer code for each diagnosis (NL - 0, MCI - 1, Dementia - 2)
    """
    if string_diagnosis == 'MCI to Dementia':
        return 2
    if string_diagnosis == 'NL':
        return 0
    if string_diagnosis == 'Dementia':
        return 2
    if string_diagnosis == 'MCI to NL':
        return 0
    if string_diagnosis == 'Dementia to MCI':
        return 1
    if string_diagnosis == 'NL to MCI':
        return 1
    if string_diagnosis == 'NL to Dementia':
        return 2
    if string_diagnosis == 'MCI':
        return 1
    return None


def run_base_experiment(features, targets, parameters, model, scoring):
    """Run an experiment with a model and calculate the mean and standard deviation for the score
    
    Args:
        features (iterable): The features to use to fit the movel
        targets (iterable): The targets to try and predict with the model
        parameters (dict): The parameter grid to search
        model (sklearn.Estimator): The sklearn estimator to instantiate
        scoring (str): The scoring method to use (see sklearn documentation)
    
    Returns:
        tuple: mean and standard deviation results for the experiment
    """
    clf = GridSearchCV(model(), parameters, scoring=scoring, cv=10, n_jobs=-1)
    clf.fit(features, targets)
    mean = clf.best_score_
    stddev = clf.cv_results_['std_test_score'][clf.best_index_]

    return mean, stddev


def main():

    modalities_to_use = ["MRI_FSX_SV_labeled_latest", "snp_labeled"]
    features, targets = get_data_for_experiment_with_modalities(modalities_to_use, "diagnosis_labeled_latest", "DX", merged_dataset, verbose=True)
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    run_base_experiment(features, targets, parameters, SVC, "f1_weighted")


    modalities_to_use = ["MRI_FSX_SV_labeled_latest", "snp_labeled"]
    features, targets = get_data_for_experiment_with_modalities(modalities_to_use, "ADAS11_labeled_latest", "ADAS11", merged_dataset, verbose=True)
    parameters = {'C':[1, 10], 'epsilon': [0.1, 0.2]}
    print(f"Will run experiment with {features.shape[0]} samples.")
    mean, stddev = run_base_experiment(features, targets, parameters, SVR, "neg_mean_squared_error")
    print(f"mean: {mean}, stddev: {stddev}")

if __name__ == "__main__":
    main()
