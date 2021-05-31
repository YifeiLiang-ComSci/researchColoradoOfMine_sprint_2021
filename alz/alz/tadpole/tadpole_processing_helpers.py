"""Helpers for TADPOLE processing
"""
import pandas as pd
import numpy as np

def extract_column_names(data, start_column_index,to_column_index, every_other=None, contains=None):
    """Extract the column names from the table given their indexes
    
    Args:
        data (pd.DataFrame): The dataframe to extract columns from
        start_column_index (int): The starting column index
        to_column_index (int): The end column index
        every_other (int): The number of items to skip and get every other value
        contains (str): The string that the column name must contain
    
    Returns:
        list: A list of strings representing the column names
    """
    if every_other is not None:
        return list(data.columns[start_column_index:to_column_index:1+every_other])
    if contains is not None:
        return [item for item in list(data.columns[start_column_index:to_column_index]) if contains in item]
    return list(data.columns[start_column_index:to_column_index])
def get_patient(data, patient_id):
    """Get data for a particular patient
    
    Args:
        data (dict): The data with modalities as keys and value as a dict of patient keys and modality values
        patient_id (str): The patient's id to look up
    
    Returns:
        dict: A dictionary with all keys from data with values for a particular patient. Will return None if key not available
    """
    patient_dict = {}
    for key in data.keys():
        try:
            patient_dict[key] = data[key][patient_id]
        except KeyError:
            patient_dict[key] = None
    return patient_dict

def get_multiple_patients(data, patient_ids):
    """Get data for multiple patients
    
    Args:
        data (dict): The data with modalities as keys and values as a dict of patient keys and modality values
        patient_ids (list): The list of patient ids to look up
    
    Returns:
        dict: A dictionary of patient disctionaries
    """
    patients_dict = {}
    for patient_id in patient_ids:
        patients_dict[patient_id] = get_patient(data, patient_id)
    return patients_dict

def get_count_in_modality(data, modality):
    """Get the number of not None modality values 
    
    Args:
        data (dict): The dataset dictionary with keys as modality and values as a dict of patient ids to modality values
        modality (string): The name of the modality in the dataset
    
    Returns:
        int: The number of values that are not None for that modality
    """
    count = 0
    for key in data[modality].keys():
        if data[modality][key] is not None:
            count +=1
    return count


def get_clean_modality(row, modality, modality_columns):
    """Get the cleaned up modality values from the data or None
    
    Args:
        row (pd.Series): The data for the modality
        modality (str): The name of the modality
        modality_columns (list): The list of strings of the columns to access
    
    Returns:
        tuple: The labeled dictionary of values or None and the vector of values or None
    """
    modality_values = row[modality_columns]
    labeled = modality_values.T.to_dict()
    vector = modality_values.to_numpy().flatten()

    ######
    ## Edge Cases
    ######

    for i, (key, value) in enumerate(labeled.items()):
        # If there are empty strings, return None
        if isinstance(vector[i], str) and not vector[i].strip():
            vector[i] = None
            labeled[key] = None
        # For each item that is a -4, convert it to None
        if isinstance(vector[i], str) and vector[i].strip() == "-4":
            vector[i] = None
            labeled[key] = None
        if isinstance(vector[i], float) and np.isnan(vector[i]):
            vector[i] = None
            labeled[key] = None
        if vector[i] == -4:
            vector[i] = None
            labeled[key] = None
        
        try:
            labeled[key] = float(value)
            vector[i] = float(vector[i])
        except:
            pass

    if all(v is None for v in vector):
        return None, None

    return labeled, vector

# Defining modality columns
# These numbers are hardcoded for the columns representing each modality
tadpole_d1_d2 = pd.read_csv("data/tadpole/TADPOLE_D1_D2.csv", low_memory=False)
tadpole_modality_columns = {
    "apoE": extract_column_names(tadpole_d1_d2, 17,18),
    "csf": extract_column_names(tadpole_d1_d2, 1902,1905),
    "cognitive_tests": extract_column_names(tadpole_d1_d2, 21,45),
    "CDRSB": extract_column_names(tadpole_d1_d2, 21,22),
    "ADAS11": extract_column_names(tadpole_d1_d2, 22,23),
    "ADAS13": extract_column_names(tadpole_d1_d2, 23,24),
    "MMSE": extract_column_names(tadpole_d1_d2, 24,25),
    "RAVLT": extract_column_names(tadpole_d1_d2, 25,29),
    "MOCA": extract_column_names(tadpole_d1_d2, 29,30),
    "Ecog": extract_column_names(tadpole_d1_d2, 30,45),
    "MRI_UCSF": extract_column_names(tadpole_d1_d2, 47,54),
    "MRI_FSL": extract_column_names(tadpole_d1_d2, 122,468),
    "MRI_FSL_SV": extract_column_names(tadpole_d1_d2, 122,468, contains="SV_"),
    "MRI_FSL_CV": extract_column_names(tadpole_d1_d2, 122,468, contains="CV_"),
    "MRI_FSL_SA": extract_column_names(tadpole_d1_d2, 122,468, contains="SA_"),
    "MRI_FSL_TA": extract_column_names(tadpole_d1_d2, 122,468, contains="TA_"),
    "MRI_FSL_TS": extract_column_names(tadpole_d1_d2, 122,468, contains="TS_"),
    "MRI_FSX": extract_column_names(tadpole_d1_d2, 486, 832),
    "MRI_FSX_SV": extract_column_names(tadpole_d1_d2, 486, 832, contains="SV_"),
    "MRI_FSX_CV": extract_column_names(tadpole_d1_d2, 486, 832, contains="CV_"),
    "MRI_FSX_SA": extract_column_names(tadpole_d1_d2, 486, 832, contains="SA_"),
    "MRI_FSX_TA": extract_column_names(tadpole_d1_d2, 486, 832, contains="TA_"),
    "MRI_FSX_TS": extract_column_names(tadpole_d1_d2, 486, 832, contains="TS_"),
    "DTI": extract_column_names(tadpole_d1_d2, 1667,1895),
    "PET_Avg": extract_column_names(tadpole_d1_d2, 18,21),
    "PET_BAI_GN": extract_column_names(tadpole_d1_d2, 838,918),
    "PET_BAI": extract_column_names(tadpole_d1_d2, 918,1105),
    "PET_AV45": extract_column_names(tadpole_d1_d2, 1174,1412),
    "PET_AV1451": extract_column_names(tadpole_d1_d2, 1414,1656),
    "PET_AV1451_UPTAKE": extract_column_names(tadpole_d1_d2, 1414,1656,every_other=1),
    "PET_AV1451_SIZE": extract_column_names(tadpole_d1_d2, 1415,1656,every_other=1),
    "demographics": extract_column_names(tadpole_d1_d2, 11,17),
    "diagnosis": extract_column_names(tadpole_d1_d2, 54,55)
}

timepoints = set(tadpole_d1_d2["VISCODE"])
tadpole_patients = set(tadpole_d1_d2["PTID"])
ordered_timepoints = sorted(list(timepoints), key = lambda x: 0 if x == "bl" else int(x[1:]))

