# +
# Data processing script to create a dense baseline dataframe for CLD.jl

import numpy as np
import pandas as pd

# +
with open("../out/fs_labels.txt") as fs_labels_file:
    fslabels = fs_labels_file.readline().strip().split(",")
with open("../out/vbm_labels.txt") as vbm_labels_file:
    vbmlabels = vbm_labels_file.readline().strip().split(",")
with open("../out/snp_labels.txt") as snp_labels_file:
    snplabels = snp_labels_file.readline().strip().split(",")
    
fs = np.load("../out/fs.npy")[:,:,0]
vbm = np.load("../out/vbm.npy")[:,:,0]
snp = np.load("../out/snp.npy")[:,:]
dx = np.load("../out/diagnoses.npy")[:,0]
# -

fs_df = pd.DataFrame(fs, columns=fslabels)
vbm_df = pd.DataFrame(vbm, columns=vbmlabels)
snp_df = pd.DataFrame(snp, columns=snplabels)
dx_df = pd.DataFrame(dx, columns=["DX"]).replace([1, .5, 0], ["HC", "MCI", "AD"])

frames = [dx_df, fs_df, vbm_df, snp_df]
final = pd.concat(frames, axis=1)

final.to_csv("../out/drshen_dx_fs_vbm_snp.csv")
print("Wrote results to ../out/drshen_dx_fs_vbm_snp.csv")
