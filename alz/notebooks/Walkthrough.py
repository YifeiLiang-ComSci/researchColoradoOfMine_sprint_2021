# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: alz
#     language: python
#     name: alz
# ---

# +
import numpy as np
import nilearn as nl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# %matplotlib inline
plt.style.use("ggplot")
# -

# # SNP Walkthrough
#
# Each SNP is a single protein in the DNA strand. The data we have is the relative change between patients and a reference DNA strand for each protein. There are `1224` proteins that are used to compare between a patient and the reference and we have `412` patients.

# **Make sure you run `python alz/data_prep.py` before going along with this walkthrough**

snp = np.load("../out/snp.npy")
snp.shape

with open("../out/snp_labels.txt") as snp_labels_file:
    snpCols = snp_labels_file.readline().split(",")
len(snpCols)

# +
# TODO
# # Get patients and use them as the row index for all following dfs
# with open("../out/patients.txt") as patients_file:
#     patients = patients_file.readlines()
# len(patients)

# +
snpDF = pd.DataFrame(snp, columns=snpCols)
snpDF.index.name="Patient"

snpDF.head()
# -

# SNP data is either 0, 1 or 2. With the normalization that becomes 0, 0.5 or 1.
#
# >**Heterozygous SNPs are encoded by 1**. For homozygous ones it is usually **0 if the individual is heterozygous for the major allele** (in the population or the one found in the reference in pairwise comparison) and **2 if it is heterozygous for the minor allele**. Although this might in fact be different in different formats. 0 and 2 are always homozygous and 1 heterozygous, though. [Source](https://biology.stackexchange.com/questions/55819/what-is-the-meaning-of-the-genotype-values-in-each-snp#comment99019_55819)

# We can also load the SNP gene relationship graph and utilize it for calculations.

G = nx.read_graphml("../out/snpGraph.graphml")

print(f"There are {len(G.nodes)} nodes in the graph and {len(list(nx.isolates(G)))} of them are not connected to any others.")

w = "r^2"
thresh = 0.2

nx.draw(G, weight=w)
plt.show()

# +
plt.figure(figsize=(20,10))

elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d[w] > thresh]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d[w] <= thresh]

# positions for all nodes
pos = nx.spring_layout(G, weight=w)
# pos = nx.random_layout(G)
# pos = nx.spectral_layout(G, weight=w)
# pos = nx.circular_layout(G)
# pos = nx.shell_layout(G)

# nodes
nodes = nx.draw_networkx_nodes(G,pos,node_size=50, edgecolors="k")

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,width=1,weight=w, edge_color="b")
nx.draw_networkx_edges(G,pos,edgelist=esmall,width=1,alpha=0.5,edge_color='b',style='dashed', weight=w)

# labels
# nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')
plt.show()
# -

cliques = list(nx.find_cliques(G))
print(f"There are {len(cliques)} cliques in the graph.")

L = nx.linalg.laplacian_matrix(G, weight=w)
L.shape

LEigs = nx.linalg.laplacian_spectrum(G, weight=w)
LEigs.shape

LEigs

WMat, WMatOrdering  = nx.attr_matrix(G, edge_attr=w)
WMat.shape

# # Brain Images
#
# VBM and Freesurfer (FS) are outputs of two types of brain scans and software. They use different atlas features.
#
# VBM uses AAL and FS uses Harvard, Tal and others.
#
# Both can be used to plot brain images using the nilearn package.
#

vbm = np.load("../out/vbm.npy")
vbm.shape

with open("../data/vbm_labels.csv") as vbm_labels_file:
    vbmCols = vbm_labels_file.readline().split(",")
len(vbmCols)

from itertools import product

vbmDF = pd.DataFrame(index=pd.MultiIndex.from_tuples(product(range(4), range(412))), columns=vbmCols)
# Fill in data
vbmDF.head()

fs = np.load("../out/fs.npy")
fs.shape

with open("../out/fs_labels.txt") as fs_labels_file:
    fsCols = fs_labels_file.readline().split(",")
len(fsCols)

fsDF = pd.DataFrame(index=pd.MultiIndex.from_tuples(product(range(4), range(412))), columns=fsCols)
# Fill in data
fsDF.head()

# # Plot FreeSurfer and VBM Brain Images from Patient 0
#
# In this section we will show how to plot the FS and VBM brain images using `alz.viz.brain.roi_map`

# +
import os

if os.path.exists("alz"):
    os.remove("alz")
    
os.symlink("../alz", "alz", target_is_directory=True) # Add a symlink in order to use the brain vizualization code
# -

from alz.viz.brain.roi_map import FSRegionOfInterestMap  # kinds of ROI brain maps.
from alz.viz.brain.roi_map import VBMRegionOfInterestMap # Import the two different

fs_labels = pd.read_csv("../data/fs_atlas_labels.csv") # Read in table of FreeSurfer atlas labels
fs_labels.head() # This table contains a translation from Dr. Shen's labels to the appropriate nilearn.atlas

patient_n = 4

patient_fs_vals = fs[patient_n]               # Get the first patient's FreeSurfer brain data
patient_BL_fs = patient_fs_vals[:,0] # Extract the Baseline (BL) column

# +
fs_roi_map = FSRegionOfInterestMap()

for i in range(0, len(patient_BL_fs)):
    preferred_atlas = fs_labels.iloc[i]["Atlas"]                         # Grab the preferred Atlas from fs_labels
    labels = fs_labels.iloc[i][preferred_atlas].split("+")               # Split regions that correspond to a single Shen lab label
    weight = patient_BL_fs[i]                                           # Get weight associated with this ROI(s)
    [fs_roi_map.add_roi(roi, weight, preferred_atlas) for roi in labels] # Add each one to the map
    
fs_roi_map.build_map(smoothed=True)  # Makes the image look nicer
fs_roi_map.plot(f"FS Patient {patient_n} (BL)") # Plots the brain image with the added ROIs
# -

vbm_labels = pd.read_csv("../data/vbm_labels.csv", header=None) # Read in the ordered vbm labels
vbm_labels.head()                                               # This table just contains the AAL atlas
                                                                # labels in the order of Dr. Shen's .xlsx
                                                                # spreadsheet.

patient_n = 4

patient_vbm_vals = vbm[patient_n]               # Get the first patient's VBM brain data
patient_BL_vbm = patient_vbm_vals[:,0] # Extract the Baseline (BL) column

# +
vbm_roi_map = VBMRegionOfInterestMap()

for i in range(0, len(patient_BL_vbm)):
    label = vbm_labels[i].iloc[0]       # Get label from list of VBM labels
    weight = patient_BL_vbm[i]         # Get weight associated with this ROI
    vbm_roi_map.add_roi(label, weight)  # Add each ROI to the brain map
    
vbm_roi_map.build_map(smoothed=True)    # Makes the image look nicer
vbm_roi_map.plot(f"VBM Patient {patient_n} (BL)")  # Plots the brain image with the added ROIs
# -


