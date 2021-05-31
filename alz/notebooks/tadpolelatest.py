# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + endofcell="--"
# # +

# +
# Data processing script to create a dense baseline dataframe for CLD.jl from the TADPOLE dataset

import pandas as pd
import pickle

with open("../out/merged_data.pkl", "rb") as pickle_file:
    data = pickle.load(pickle_file)

# # +
modalities_to_use = ["diagnosis_latest",
"snp_labeled",
"MRI_FSX_SV_labeled_latest",
"MRI_FSX_CV_labeled_latest",
"MRI_FSX_SA_labeled_latest",
"MRI_FSX_TA_labeled_latest",
"MRI_FSX_TS_labeled_latest",
"MRI_FSL_SV_labeled_latest",
"MRI_FSL_CV_labeled_latest",
"MRI_FSL_SA_labeled_latest",
"MRI_FSL_TA_labeled_latest",
"MRI_FSL_TS_labeled_latest",
]

""" Column 'ST8SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16' had 389 out of 733 rows with N/A. 
Column 'ST8SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16' had 275 out of 733 rows with N/A. 
36 other columns had all N/As (so they are also removed in step below, and the remaining columns all had less than 200 N/A's)
By dropping these two columns, we go from 270 rows after removing N/A's to 524 rows
"""
drop = ["ST8SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16", "ST8SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16"]#"FAQ", "MOCA", "CDRSB", "ST8SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16"]#, "ADAS11", "ADAS13", "MMSE"]


# -

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

    df = pd.concat([pd.DataFrame.from_records(data[modality]).T for modality in modalities], axis=1)
    
    #The first column is labeled with an integer, 0. Replace with 'diagnosis'
    new_names = list(df.columns)
    new_names[0] = "diagnosis"
    df.columns = new_names
    
    ecog_columns = [column for column in df.columns if column.startswith("Ecog")]
    
        
    df = df.drop(ecog_columns, axis=1)
    
    if columns_to_drop is not None:
        df = df.drop(columns_to_drop, axis=1)

    if verbose:
        print(df.shape)
        print("Row analysis of NaNs")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
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

        """
    targets = df[[target_column]]
    if target_column == "DX":
        targets = targets["DX"].apply(convert_string_to_integer_diagnosis)
        """
    

    #features = df.drop(target_column, axis=1)

    return df 

features = get_data_for_experiment_with_modalities(modalities_to_use, None, None, data, drop, verbose=True)

# Lengths obtained by reading each modality in individually and removing the necessary columns
# # +
diagnosis_latest_LEN = 1
snp_labeled_LEN = 1224
MRI_FSX_SV_labeled_latest_LEN = 44
MRI_FSX_CV_labeled_latest_LEN = 71
MRI_FSX_SA_labeled_latest_LEN = 72
MRI_FSX_TA_labeled_latest_LEN = 70
MRI_FSX_TS_labeled_latest_LEN = 70
MRI_FSL_SV_labeled_latest_LEN = 44
MRI_FSL_CV_labeled_latest_LEN = 71
MRI_FSL_SA_labeled_latest_LEN = 72
MRI_FSL_TA_labeled_latest_LEN = 70
MRI_FSL_TS_labeled_latest_LEN = 70

features.to_csv("../out/tadpole_dx_snp_fsx_fsl_latest.csv")
print("Wrote results to ../out/tadpole_dx_snp_fsx_fsl_latest.csv")
