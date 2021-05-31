# Repository Organization

Welcome to the portal for generating usable to be used in MInDS@Mines Alzhiemer's projects!

The organization of this repository is as follows:
 - `/alz/drshen` contains scripts to parse the `.xlsx` files given to our group by Dr. Li Shen
 - `/alz/tadpole` contains scripts to parse the `.csv` files provided by the [TADPOLE project](https://tadpole.grand-challenge.org/)
 - `/alz/viz` contains scripts to visualize results on the brain
 - `/notebooks` contains various walkthroughs related to the `alz` repository


## Data Access

We're using Google Cloud Storage for data access. Once you are added to the minds-mines project on GCP, and have installed the [Google Cloud SDK](https://cloud.google.com/sdk) you can run the following command to get data added in the data folder:

```bash
gsutil cp -r 'gs://minds-data/alzheimers/adni/*' data/
```

## Setting up the Project

Similar to all of our [Python-based projects](https://gitlab.com/minds-mines/minds-mines#python) we create our environment using Pipenv: 

```bash
pipenv install
pipenv shell
```

## Generating Alzheimer's Data Matrices from TADPOLE data

The current usage for the tadpole data is through the following 5 files in the `alz/tadpole` folder:

1. `tadpole_processing.py` - processes the `data/CanSNPs_Top40Genes_org.xlsx` file and the `data/tadpole/TADPOLE_D1_D2.csv` file and outputs a pickled dictionary of all the data to `out/merged_data.pkl`
2. `tadpole_processing_helpers.py` - contains helper functions used for processing and analyzing the data as well as the lookups for **Modality to excel sheet columns**, **timepoints**, and **patient IDs**
3. `tadpole_analysis.py` - contains analysis that is applied to the data to view heatmaps of the modality overlap over timepoints as well as how many samples we have with n timepoints.
4. `tadpole_testing.py` - a helper for testing and investigating the output of the processed data. Useful with a debugger to investigate NaNs within the data.
5. `tadpole_base_experiment.py` - data cleaning helpers and example experiment with gridsearch outputing mean and standard deviation results. I recommend you use these functions for your own experiments with this dataset.

## Generating Alzheimer's Data Matrices from Dr. Li Shen's data

In order to generate the Alzheimer's Disease data matrices run the following
after `unzipping` the `xlsx` data files contained in `data/`.

```
python alz/drshen/data_prep.py
```

## Visualization Tools

In this section we provide minimum working examples on the the visualization tools available. When creating figures for a paper you will modify the scripts for your own use case.

### Visualzing TADPOLE Data on a Brain

```bash
python alz/viz/brain/tadpole_map_test.py
```

### Visualzing Dr. Li Shen's Data on a Brain

```bash
python alz/viz/brain/shen_map_test.py
```

## Data Procedure / Changes

### Changes to .xlsx
1. Changed MMSE_SL sheet name to MMSE_BL
2. Changed sheet names in VBM_mod to look like other sheet names
3. Changed FLUENCY to FLU in .xlsx

### Data
1. Removed `018_S_0055` from `snp_final.xlsx`, entry didn't exist in `longitudinal basic infos.xlsx`

### Acknowledging the TADPOLE/ADNI Data (Conference)

Data collection and sharing for this project was funded by the Alzheimer’s Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). A full list of funding sources for ADNI is provided in the document 'Alzheimers Disease Neuroimaging Initiative (ADNI) Data Sharing and Publication Policy' available through adni.loni.usc.edu/.

This work uses the TADPOLE data sets https://tadpole.grand-challenge.org constructed by the EuroPOND consortium http://europond.eu funded by the European Unions Horizon 2020 research and innovation programme under grant agreement No 666992.


### Acknowledging the TADPOLE/ADNI Data (Journal)

Data collection and sharing for this project wasfunded by the Alzheimer’s Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from the following: AbbVie, Alzheimer’s Association; Alzheimer’s Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research \& Development, LLC.; Johnson \& Johnson Pharmaceutical Research \& Development LLC.; Lumosity; Lundbeck; Merck \& Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics. The Canadian Institutes of Health Research is providing funds to support ADNI clinical sites in Canada. Private sector contributions are facilitated by the Foundation for the National Institutes of Health (www.fnih.org). The grantee organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer’s Therapeutic Research Institute at the University of Southern California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California. 
This work uses the TADPOLE data sets https://tadpole.grand-challenge.org constructed by the EuroPOND consortium http://europond.eu funded by the European Unions Horizon 2020 research and innovation programme under grant agreement No 666992.
