import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import alz.tadpole.tadpole_processing_helpers as helpers
except ImportError:
    import tadpole_processing_helpers as helpers
import pickle
import os.path

pickle_path = "out/merged_data.pkl"

with open(pickle_path, "rb") as pickle_file:
    merged_dataset = pickle.load(pickle_file)

def check_for_example(key, data, n=5):
    """Find an example patient
    
    Args:
        key (str): The key to check in the dataset
        data (dict): The dataset to search through
        n (int): The number of example patients to check for (Defaults to 5)
    
    Returns:
        dict/str: The first n not None patient found or a string error saying all are None
    """
    patients_to_return = []
    for patient in data[key].keys():
        if data[key][patient] is not None:
            patients_to_return.append((patient, data[key][patient]))
        if len(patients_to_return) == n:
            return patients_to_return
    if len(patients_to_return) > 1:
        return patients_to_return
    return "All patients are none"

keys_that_are_all_none = []
for key in merged_dataset.keys():
    print(key)
    examples = check_for_example(key, merged_dataset)
    print(examples)
    if isinstance(examples, str) and examples == "All patients are none":
        keys_that_are_all_none.append(key)

print(f"The following keys are all None:")
for key in keys_that_are_all_none:
    print(key)

# Checking MRI_FSX
modality = "MRI_FSX_SV"
columns = helpers.tadpole_modality_columns[modality]
print(helpers.tadpole_d1_d2[columns].head())
