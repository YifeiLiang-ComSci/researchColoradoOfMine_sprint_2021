# ROI Map Test
import random
import pandas
import numpy as np

from clean_labels import VBMDataCleaner # TODO: Depricate VBMDataCleaner and add labels to gcloud
from roi_map import VBMRegionOfInterestMap
from roi_map import FSRegionOfInterestMap

# The following .csv contains various nilearn.atlas ROI names
# paired with a specific type of atlas you can nilearn.fetch.
# A corresponding "perferred" atlas is also provided.
fs_labels = pandas.read_csv("alz/viz/brain/fs_atlas_labels.csv")

# Removes starting and trailing characters from a list of labels
# as provided by Dr. Shen's Lab
vbm_cleaner = VBMDataCleaner()
vbm_cleaner.load_data("data/longitudinal imaging measures_VBM_mod_final.xlsx")
vbm_labels = vbm_cleaner.clean()

# Create the corresponding (collapsed coefficent)
# vector that contains the ordered feature weights. 
#
# For simplicity. I am just creating a random feature vector.
# TODO: Show an example with an individual patient from data/.
fs_features = random.sample(range(100), len(fs_labels))
vbm_features = random.sample(range(100), len(vbm_labels))

# Create FreeSurfer ROI Map
fs_roi_map = FSRegionOfInterestMap()
for index, row in fs_labels.iterrows():
    atlas = row["Atlas"]
    # Since some labels encompass multiple brain areas they are separated with a + in fs_atlas_labels.csv
    # This is a janky solution. TODO: Fix this.. Or just upload to gcloud and not care.
    rois = row[atlas].split("+")
    [fs_roi_map.add_roi(roi, fs_features[index], atlas) for roi in rois]

fs_roi_map.build_map(smoothed=True)
fs_roi_map.plot("FreeSurfer")
fs_roi_map.save("out/fs_fig.png", "FreeSurfer")

# Create VBM ROI Map
vbm_roi_map = VBMRegionOfInterestMap()
# TODO: use a dataframe (derived from a .csv uploaded to gcloud) 
# that looks similar to the FSRegionOfInterestMap implementation.
for label, weight in zip(vbm_labels, vbm_features):
    vbm_roi_map.add_roi(label, weight)

vbm_roi_map.build_map(smoothed=True)
vbm_roi_map.plot("VBM")
vbm_roi_map.save("out/vbm_fig.png", "VBM")
