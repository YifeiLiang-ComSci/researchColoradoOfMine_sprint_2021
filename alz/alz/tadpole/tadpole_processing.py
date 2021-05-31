"""This file walks through processing, visualizing and understanding the TADPOLE and ADNI Dataset
"""

# In the `data` folder you should have the following files:
# `CanSNPs_Top40Genes_org.xlsx` - The SNP gene data for each patient id
# `tadpole/TADPOLE_D1_D2.csv` - The multimodal brain scans for each patient at various times

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import alz.notebooks.tadpole_processing_helpers as helpers
except ImportError:
    import tadpole_processing_helpers as helpers
import pickle
import os.path

from tadpole_processing_helpers import tadpole_modality_columns

make_pickle = False
pickle_path = "out/merged_data.pkl"
only_snp_patients = True

if not os.path.exists(pickle_path) or make_pickle:

    # Read SNP data 
    snp_data = pd.read_excel("data/CanSNPs_Top40Genes_org.xlsx", sheet_name="CanSNPs_Top40Genes_org", header=0)
    snp_patients = set(snp_data["ID"])
    snp_n_patients, n_genes = snp_data.shape # 733, 1225
    n_genes -=1 # Don't count id column
    print(f"We have {snp_n_patients} patients with gene information about {n_genes} genes.")
    assert len(snp_patients) == snp_n_patients

    # Read TADPOLE data
    tadpole_d1_d2 = helpers.tadpole_d1_d2
    timepoints = helpers.timepoints
    tadpole_patients = helpers.tadpole_patients

    snp_and_tadpole_patients = snp_patients.intersection(tadpole_patients)
    print(f"{len(snp_and_tadpole_patients)} patients overlap across the {snp_n_patients} with SNP data and the {len(tadpole_patients)} with TADPOLE data.")

    if only_snp_patients:
        tadpole_d1_d2 = tadpole_d1_d2[tadpole_d1_d2["PTID"].isin(snp_patients)]
        tadpole_patients = snp_patients
        print("Only using SNP patients - I recommend you change the pickle path so that you know you're only working with SNP patients.")


    # ## Ideal dictionary setup:
    # my_data = {
    #     "Modality_Timepoint": {
    #         "Patient ID": [] # Patient vector
    #     },
    #     "Modality_Latest": {
    #         "Patient ID": [] # Patient vector
    #     }
    # }
    # ## Usage
    # # To get a modality
    # pd.Series(my_data["VBM_M16"]).to_numpy()
    # # To get a patient
    # helpers.get_patient(my_data, "011_S_0003")
    # # To get multiple patients
    # helpers.get_multiple_patients(my_data, ["011_S_0003"])



    # Initialize the merged_dataset dictionary
    merged_dataset = {
        "snp_labeled": {}, # Dictionary of gene values
        "snp": {} # Numpy array of gene values
    }
    for patient in tadpole_patients:
        merged_dataset["snp_labeled"][patient] = None
        merged_dataset["snp"][patient] = None

        for modality in tadpole_modality_columns.keys():
            for timepoint in timepoints.union(set(["latest", "latest_timepoint"])):
                modality_timepoint_key_name = f"{modality}_{timepoint}"
                labeled_modality_timepoint_key_name = f"{modality}_labeled_{timepoint}"

                if modality_timepoint_key_name not in merged_dataset.keys():
                    merged_dataset[modality_timepoint_key_name] = {}
                if labeled_modality_timepoint_key_name not in merged_dataset.keys():
                    merged_dataset[labeled_modality_timepoint_key_name] = {}

                merged_dataset[modality_timepoint_key_name][patient] = None
                merged_dataset[labeled_modality_timepoint_key_name][patient] = None


    # Loop over SNP patients and add them
    for patient in snp_patients:
        snp_patient_data = snp_data.query(f"ID == '{patient}'").drop("ID", axis=1).T
        # Dictionary of gene values
        merged_dataset["snp_labeled"][patient] = snp_patient_data.to_dict()[snp_patient_data.T.index[0]]
        # Flattened Numpy array of gene values
        merged_dataset["snp"][patient] = snp_patient_data.to_numpy().flatten()


    # Loop over Tadpole data and add it
    for row_num, row in tadpole_d1_d2.iterrows():
        row_viscode = row["VISCODE"]
        patient = row["PTID"]
        for modality, modality_columns in tadpole_modality_columns.items():
            merged_dataset[f"{modality}_labeled_{row_viscode}"][patient], merged_dataset[f"{modality}_{row_viscode}"][patient] = helpers.get_clean_modality(row, modality, modality_columns)

    ordered_timepoints = helpers.ordered_timepoints
    for patient in tadpole_patients:
        for modality in tadpole_modality_columns.keys():
            for timepoint in reversed(ordered_timepoints):
                if  merged_dataset[f"{modality}_latest"][patient] is None and merged_dataset[f"{modality}_{timepoint}"][patient] is not None:
                    merged_dataset[f"{modality}_latest"][patient] = merged_dataset[f"{modality}_{timepoint}"][patient]
                    merged_dataset[f"{modality}_labeled_latest"][patient] = merged_dataset[f"{modality}_labeled_{timepoint}"][patient]
                    merged_dataset[f"{modality}_latest_timepoint"][patient] = timepoint
                    merged_dataset[f"{modality}_labeled_latest_timepoint"][patient] = timepoint

    print(f"The following modalities are available in the dataset: {merged_dataset.keys()}")


    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(merged_dataset, pickle_file)


else:

    with open(pickle_path, "rb") as pickle_file:
        merged_dataset = pickle.load(pickle_file)

    print(merged_dataset.keys())

