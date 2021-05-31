# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

# loads in ADN1 SNP data into dataframe, renaming 'ID' to 'PTID' so it matches naming on TADPOLE data
df = pd.read_csv("alz/data/CanSNPs_Top40Genes_org.csv")
df = df.rename(columns={'ID': 'PTID'})
df.head()

# reads in TADPOLE data file
tadpole = pd.read_csv("alz/data/tadpole/TADPOLE_D1_D2.csv")

# creates new dataframe from TADPOLE dataframe of only relevant columns 
tp = tadpole[['RID', 'PTID', 'VISCODE']]
tp.head()

# merge the TADPOLE and SNP dataframes based on PTID, sorts dataframe by RID number
out = tp.merge(df, how='left', on='PTID') 
out = out.sort_values(by=['RID', 'VISCODE'])
out = out.reset_index(drop=True)
out.head()

# create lists of Patients (RID), Timesteps (VISCODES), and Features (SNPNames)
# these lists will then be used as input for 3D array
Patients = list((out['RID'].unique()))
tags = np.load("alz/out/SNP/snptags.npy")
Viscodes = list(tags[1]) # accesses list of viscodes from SNP tags file
SNPNames = list([col for col in out if col.startswith('rs')])

# create empty numpy array, with its shape being defined as the length of each dimension
# fill each dimension with 'NaN' value
snp_dat = np.full(shape=(len(Patients), len(Viscodes), len(SNPNames)), fill_value=np.nan)  


# function that returns a list of viscodes that each patient has
def patientViscodes (patient):
    patientViscodes = list(out[out["RID"] == patient]["VISCODE"].values)
    return patientViscodes


# iterates through each dimension of the 3D array. 
# adds SNP values if the patient has data from a VISCODE, at the correct patient and viscode. 
for i in range(0, len(Patients)):
    patientVIS = patientViscodes(Patients[i])
    for j in range(0, len(Viscodes)):
        if (Viscodes[j] in patientVIS):
            snp_dat[i, j,:] = out[(out["RID"] == Patients[i]) & (out["VISCODE"] == Viscodes[j])][SNPNames]

np.save("alz/out/SNP/finished_snp_dat.npy", snp_dat)
