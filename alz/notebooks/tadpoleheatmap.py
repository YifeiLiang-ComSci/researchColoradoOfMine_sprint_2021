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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_data_and_labels(partial_filename): 
    data = np.load(partial_filename + "_data.npy", allow_pickle=True)
    labels = np.load(partial_filename + "_labels.npy", allow_pickle=True)
    return data, labels


# +
cog_data, cog_labels = load_data_and_labels("../out/tadpole_npy/cognitive_tests/cog")
csf_data, csf_labels = load_data_and_labels("../out/tadpole_npy/csf/csf")
dem_data, dem_labels = load_data_and_labels("../out/tadpole_npy/demographics/dem")
dx_data, dx_labels = load_data_and_labels("../out/tadpole_npy/diagnosis/dx")
snp_data, snp_labels = load_data_and_labels("../out/tadpole_npy/genetic/snp")

# MRI Groups
dti_data, dti_labels = load_data_and_labels("../out/tadpole_npy/mri/dti")
mri_lfs_data, mri_lfs_labels = load_data_and_labels("../out/tadpole_npy/mri/mri_lfs")
mri_ucsf_data, mri_ucsf_labels = load_data_and_labels("../out/tadpole_npy/mri/mri_ucsf")
mri_xfs_data, mri_xfs_labels = load_data_and_labels("../out/tadpole_npy/mri/mri_xfs")

# PET Groups
pet_av1451_data, pet_av1451_labels = load_data_and_labels("../out/tadpole_npy/pet/pet_av1451")
pet_av45_data, pet_av45_labels = load_data_and_labels("../out/tadpole_npy/pet/pet_av45")
pet_avg_data, pet_avg_labels = load_data_and_labels("../out/tadpole_npy/pet/pet_avg")
pet_bai_data, pet_bai_labels = load_data_and_labels("../out/tadpole_npy/pet/pet_bai")


# -

def draw_longitudinal_heatmap_for_modality(_fig, _data, _label, _times, row):
    for i in range(0, len(_times)):
        axn = _fig.add_subplot(11, 22, i+1+(row*len(times)))
        heat_vals = np.copy(_data[:,i,:])
        heat_vals[0,0] = 0

        hmap = sns.heatmap(heat_vals, ax=axn, cbar=False, xticklabels=False, yticklabels=False)
        if row == 0:
            hmap.set_title(_times[i])
        if i == 0:
            hmap.set_ylabel(_label)
            
    return row+1


def encode_categorical(_data):
    df = pd.Categorical(_data.flatten())
    encoded = df.codes.reshape(_data.shape)
    copy = np.copy(encoded).astype(float)
    copy[copy==-1]=np.nan
    return copy 


# +
fig = plt.figure(figsize=(44, 22))
times = cog_labels[1]

i=0
i=draw_longitudinal_heatmap_for_modality(fig, snp_data[:,:,0:20], "SNP", times, i)
#i=draw_longitudinal_heatmap_for_modality(fig, csf_data.astype(float), "CSF", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, encode_categorical(dx_data), "Diagnoses", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, cog_data, "Cognitive Scores", times, i)
#i=draw_longitudinal_heatmap_for_modality(fig, dem_data, "Demographics", times, i)
 
# MRI Groups
i=draw_longitudinal_heatmap_for_modality(fig, dti_data.astype(float)[:,:,0:20], "DTI", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, mri_lfs_data.astype(float)[:,:,0:20], "MRI LFS", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, mri_ucsf_data.astype(float)[:,:,0:20], "MRI UCSF", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, mri_xfs_data.astype(float)[:,:,0:20], "MRI XFS", times, i)

# PET Groups
i=draw_longitudinal_heatmap_for_modality(fig, pet_av1451_data.astype(float)[:,:,0:20], "PET AV1451", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, pet_av45_data.astype(float)[:,:,0:20], "PET AV45", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, pet_avg_data.astype(float), "PET AVG", times, i)
i=draw_longitudinal_heatmap_for_modality(fig, pet_bai_data.astype(float)[:,:,0:20], "PET BAI", times, i)
# -

# # Build baseline X and Y data matrices and save as `.csv`

choose_BL = 0

dx_df         = pd.DataFrame(dx_data[:,choose_BL,:], columns=dx_labels[2])
dem_df        = pd.DataFrame(dem_data[:,choose_BL,:], columns=dem_labels[2])
snp_df        = pd.DataFrame(snp_data[:,choose_BL,:], columns=snp_labels[2])
mrilfs_df     = pd.DataFrame(mri_lfs_data[:,choose_BL,:], columns=mri_lfs_labels[2])
mrilfs_df_fix = mrilfs_df.drop("ST100SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16", axis=1) # Perhaps an off-by-one-error.
# mriucsf_df  = pd.DataFrame(mri_ucsf_data[:,choose_BL,:], columns=mri_ucsf_labels[2]) # Won't use since it is only 7 columns.
mrixfs_df     = pd.DataFrame(mri_xfs_data[:,choose_BL,:], columns=mri_xfs_labels[2])
mrixfs_df_fix = mrixfs_df.drop("ST100SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16", axis=1) # Perhaps another off-by-one-error.
petbai_df     = pd.DataFrame(pet_bai_data[:,choose_BL,:], columns=pet_bai_labels[2])

frames=[dx_df, dem_df, mrilfs_df_fix, mrixfs_df_fix, snp_df]

full = pd.concat(frames, axis=1)
full_dc = full.dropna(axis=1, thresh=700)
dense = full_dc.dropna()
final = dense[dense.DX != "Dementia to MCI"]

final.to_csv("../out/tadpole_dx_dem_lfs_xfs_snp_BL.csv")


