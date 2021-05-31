"""Analyze the processed data for tadpole
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os.path
try:
    import alz.tadpole.tadpole_processing_helpers as helpers
    from alz.tadpole.tadpole_processing_helpers import tadpole_modality_columns, ordered_timepoints, tadpole_patients
except ImportError:
    import tadpole_processing_helpers as helpers
    from tadpole_processing_helpers import tadpole_modality_columns, ordered_timepoints, tadpole_patients

pickle_path = "out/merged_data.pkl"

with open(pickle_path, "rb") as pickle_file:
    merged_dataset = pickle.load(pickle_file)


# Modality counter
modality_timepoint_counts = np.zeros((len(tadpole_modality_columns), len(ordered_timepoints)))
unlabeled_modalities = [modality for modality in list(tadpole_modality_columns.keys()) if "labeled" not in modality]

i = 0
for modality in unlabeled_modalities:
    j = 0
    for timepoint in ordered_timepoints:
        modality_timepoint_counts[i, j] = helpers.get_count_in_modality(merged_dataset, f"{modality}_{timepoint}")
        j+=1
    i+=1

fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(modality_timepoint_counts, annot=True, yticklabels=unlabeled_modalities, xticklabels=ordered_timepoints, cmap="coolwarm", ax=ax, fmt="g")
plt.tight_layout()
plt.savefig("out/modality_counts.png")
plt.clf()

print("Saving modality counts in out/modality_counts.png")

# Patients count per n scans
any_modality = "diagnosis_labeled_m66"
patient_counts = {}
for patient in merged_dataset[any_modality].keys():
    patient_counts[patient] = {}
    for modality in unlabeled_modalities:
        modality_counter = 0
        for timepoint in ordered_timepoints:
            if merged_dataset[f"{modality}_{timepoint}"][patient] is not None:
                modality_counter +=1
        patient_counts[patient][modality] = modality_counter


patient_counts_df = pd.DataFrame(patient_counts).T
modality_n_counter = np.zeros((len(tadpole_modality_columns), len(ordered_timepoints)+1))
for i,modality in enumerate(unlabeled_modalities):
    n_counter = pd.value_counts(patient_counts_df[modality])
    for index_number in n_counter.index:
        modality_n_counter[i,index_number] = n_counter.get(index_number)

# Flip the matrix to do a reverse cumulative sum
# People with 4 scans should also count for 3 scans
modality_n_counter = np.flip(np.flip(modality_n_counter, axis=1).cumsum(axis=1), axis=1)

fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(modality_n_counter, annot=True, yticklabels=unlabeled_modalities, xticklabels=[str(v) for v in range(len(ordered_timepoints)+1)], cmap="coolwarm", ax=ax, fmt="g")
plt.tight_layout()
plt.savefig("out/modality_n_counter.png")
plt.clf()

print("Saving modality counter in out/modality_n_counter.png")
