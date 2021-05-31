from clean_labels import VBMDataCleaner
from nilearn import datasets

cleaner = VBMDataCleaner()
cleaner.load_data("data/longitudinal imaging measures_VBM_mod_final.xlsx")
cleaned = cleaner.clean()

aal_atlas = datasets.fetch_atlas_aal("SPM12")

print("Len: " + str(len(cleaned)))

for x in cleaned:
    print(str(aal_atlas.labels.index(x)) + " " + x)
