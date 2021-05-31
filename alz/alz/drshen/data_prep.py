""" data_prep.py
Concatenate all modalities and drop any missing values' rows then normalize and split up and save as .npy files

All files get saved in the /out directory.
"""

from sklearn import preprocessing
import pandas as pd
import numpy as np
import networkx as nx
import csv

TIMES = ["BL", "M6", "M12", "M18", "M24", "M36"]

# Remove unwanted TIMES, INPUT_LABELS or OUTPUT_LABELS
# VBM modified instead, SNP Data constant along time dimension (talk w/ Hua)
# Only need to incorperate a few to show validiity of new method
#TIMES.remove("M6")
#TIMES.remove("M12")
TIMES.remove("M18")
#TIMES.remove("M24")
TIMES.remove("M36")

print("TIMES being parsed: " + ", ".join(TIMES))

def reorder_snp(basic_data, snp_data):
    """Reorder the SNP Data to be correctly associated with the other .xlsx data frames"""
    tmp = snp_data['ID'].apply(lambda x: x[-4:])
    tmp.rename('sortID', inplace=True)

    snp_data = pd.concat([tmp, snp_data], axis=1)
    snp_data.sort_values(by='sortID', inplace=True)
    snp_data.drop(columns=['sortID'], inplace=True)

    all_patients = basic_data["SubjID"]
    snp_final = pd.merge(all_patients.to_frame(), snp_data, left_on='SubjID', right_on='ID', how='outer')

    # Remove 018_S_0055 (because it doesn't exist in longitudinal data
    snp_final = snp_final[snp_final.ID != '018_S_0055']

    # Remove unwanted ID labels
    snp_final.drop(columns=['SubjID', 'ID'], inplace=True)
    return snp_final

def drop_na_and_normalize(fs, vbm, snp, scores, diagnoses):
    """Drop rows that contain any NaN Values @ any TIME"""
    concatenated = snp
    for f, v, s, d in zip(fs, vbm, scores, diagnoses):
        concatenated = pd.concat([concatenated, f, v, s, d], axis=1)
    na_dropped = concatenated.dropna() 
    # Normalize the data
    norm_na_dropped = (na_dropped-na_dropped.min())/(na_dropped.max()-na_dropped.min())

    snp_dropped = norm_na_dropped[snp.columns]
    fs_dropped, vbm_dropped, scores_dropped, diagnoses_dropped = [], [], [], []
    
    for f, v, s, d in zip(fs, vbm, scores, diagnoses):
        fs_dropped.append(norm_na_dropped[f.columns])
        vbm_dropped.append(norm_na_dropped[v.columns])
        scores_dropped.append(norm_na_dropped[s.columns])
        diagnoses_dropped.append(norm_na_dropped[d.name])

    return np.transpose(np.stack(fs_dropped, axis=0), (1, 2, 0)), np.transpose(np.stack(vbm_dropped, axis=0), (1, 2, 0)), np.array(snp_dropped), np.transpose(np.stack(scores_dropped, axis=0),(1, 2, 0)), np.transpose(np.stack(diagnoses_dropped, axis=0))

# Load input data
freesurfer_data = []
vbm_data = []
cognitive_scores = []
cognitive_diagnoses = []

data_path = "data"

info_path = data_path + "/longitudinal basic infos.xlsx"
fs_path = data_path + "/longitudinal imaging measures_FS_final.xlsx"
vbm_path = data_path + "/longitudinal imaging measures_VBM_mod_final.xlsx"
snp_path = data_path + "/CanSNPs_Top40Genes_org.xlsx"
ravlt_path = data_path + "/longitudinal cognitive scores_RAVLT.xlsx"
diagnosis_path = data_path + "/longitudinal diagnosis.xlsx"

snp_data = pd.read_excel(snp_path, sheet_name="CanSNPs_Top40Genes_org", header=0)
snp_data_labels = list(snp_data.columns)[1:] # Get rid of the ID label
info_data = pd.read_excel(info_path, sheet_name="Sheet1", header=0)

snp_data = reorder_snp(info_data, snp_data)

for time in TIMES:
    freesurfer_data.append(pd.read_excel(fs_path, sheet_name="FS_" + time, header=0)) 
    vbm_data.append(pd.read_excel(vbm_path, sheet_name="VBM_mod_" + time, header=0)) 

    cognitive_scores.append(pd.read_excel(ravlt_path, sheet_name="RAVLT_"+time, header=0))
    cognitive_diagnoses.append(pd.read_excel(diagnosis_path, header=0)[time+"_DX"])

freesurfer_data, vbm_data, snp_data, cognitive_scores, cognitive_diagnoses = drop_na_and_normalize(freesurfer_data, vbm_data, snp_data, cognitive_scores, cognitive_diagnoses)

np.save("out/fs.npy", freesurfer_data)
print(f"Saved Free Surfer Data with shape {freesurfer_data.shape} as fs.npy")
np.save("out/vbm.npy", vbm_data)
print(f"Saved VBM Data with shape {vbm_data.shape} as vbm.npy")
np.save("out/snp.npy", snp_data)
print(f"Saved SNP Data with shape {snp_data.shape} as snp.npy")
np.save("out/ravlt.npy", cognitive_scores)
print(f"Saved Cognitive Scores Data with shape {cognitive_scores.shape} as ravlt.npy")
np.save("out/diagnoses.npy", cognitive_diagnoses)
print(f"Saved Cognitive Diagnoses Data with shape {cognitive_diagnoses.shape} as diagnoses.npy")

vbm_labels = pd.read_excel(vbm_path, sheet_name="VBM_mod_BL", header=0)
fs_labels = pd.read_excel(fs_path, sheet_name="FS_BL", header=0)

with open("out/vbm_labels.txt", "w") as vbm_csv:
    wr = csv.writer(vbm_csv)
    wr.writerows([list(vbm_labels)])
print("Saved VBM Labels to vbm_labels.txt")

with open("out/fs_labels.txt", "w") as fs_csv:
    wr = csv.writer(fs_csv)
    wr.writerows([list(fs_labels)])
print("Saved Free Surfer Labels to fs_labels.txt")

with open("out/snp_labels.txt", "w") as snp_labels:
    wr = csv.writer(snp_labels)
    wr.writerows([snp_data_labels]) 
print("Saved SNP Labels to snp_labels.txt")


# SNP Graph
snpLD = pd.read_excel("data/CanSNPs_Top40Genes_org.xlsx", sheet_name="CanSNPs_Top40Genes_LD", index_col=[0, 1])
# Get the dictionary of dictionaries where each node has a dictionary of nodes it is connected to with metrics and their values 
# ex. {"a": {"b": {"dist": 12, "weight": 17}, "c": {"dist": 10, "weight": 42}}}
dods = snpLD.drop("T-int", axis=1).groupby(level=0).apply(lambda snpLD: snpLD.xs(snpLD.name).to_dict("index")).to_dict()
G = nx.from_dict_of_dicts(dods)

addedNodes = set(snpLD.index.get_level_values(0)).union(set(snpLD.index.get_level_values(1)))
remainingNodes = set(snp_data_labels).difference(addedNodes)
G.add_nodes_from(remainingNodes)

nx.write_graphml(G, "out/snpGraph.graphml")
print("Saved SNP Graph to snpGraph.graphml")

