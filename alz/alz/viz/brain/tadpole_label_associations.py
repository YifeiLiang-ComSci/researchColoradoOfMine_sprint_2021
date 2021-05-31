import os
import pickle
import pandas as pd
from nilearn import datasets
import re
import csv

#  This script generates the modality files (FSL_CV_atlas_labels.csv, FSX_CV_atlas_labels.csv, ect.).
#  The script takes the associations found in description_associations.csv and generates the files from those.
#  In the first column of the description_associations.csv file there's a description of the brain region 
#  taken from the TADPOLE_D1_D2_Dict.csv file.
#  The Harvard_Oxford AAL and Talairach columns contain the labels in those corresponding atlases
#  though the Harvard_Oxford labels don't seem to work.
#  The atlas column has the name of the prefered atlas for the brain region.

#  If these associations need to be changed, modify the label in the description_associations.csv file
#  and then run this script.


#  create complete label file writes all of the labels from the atlases to a file for easier searching
def create_complete_label_file():
    harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    aal_atlas = datasets.fetch_atlas_aal("SPM12")
    talairach_atlas = datasets.fetch_atlas_talairach("ba")

    file = open('alz/viz/brain/all_atlas_labels.txt', 'w')
    for label in harvard_oxford_atlas.labels:
        if label is "Background":
            continue
        file.write(f"Harvard-Oxford: {label}\n")

    for label in aal_atlas.labels:
        file.write(f"AAL: {label}\n")

    for label in talairach_atlas.labels:
        file.write(f"Talairach: {label}\n")

    file.close()



def create_atlas_map(scanType):
    #  load data to associate 
    tadpole_d1_d2_dict = pd.read_csv("data/tadpole/TADPOLE_D1_D2_Dict.csv", low_memory=False)

    #  load data from description associations
    description_associations = pd.read_csv("alz/viz/brain/description_associations.csv")

    dataframe_columns = []
    for column in tadpole_d1_d2_dict:
        dataframe_columns.append(column)

    # I don't think this is neccessary but at this point I don't think it's really hurting anything
    tadpole_dataframe = pd.DataFrame(tadpole_d1_d2_dict, columns = dataframe_columns)

    #  create dataframe with either fsl or fsx data
    tadpole_data = tadpole_dataframe.loc[tadpole_dataframe['TBLNAME'] == ('UCSF' + scanType)]
    
    #  list of files that will contain atlas association information
    target_files = {"SV" : "alz/viz/brain/" + scanType + "_atlas_labels/" + scanType + "_SV_atlas_labels.csv",
                    "CV" : "alz/viz/brain/" + scanType + "_atlas_labels/" + scanType + "_CV_atlas_labels.csv",
                    "SA" : "alz/viz/brain/" + scanType + "_atlas_labels/" + scanType + "_SA_atlas_labels.csv",
                    "TA" : "alz/viz/brain/" + scanType + "_atlas_labels/" + scanType + "_TA_atlas_labels.csv",
                    "TS" : "alz/viz/brain/" + scanType + "_atlas_labels/" + scanType + "_TS_atlas_labels.csv"}

    #  create header for all files
    for fileKey in target_files:
        with open (target_files[fileKey], mode='w') as target_file:
            csv_writer = csv.writer(target_file, delimiter=',')
            csv_writer.writerow(["TADPOLE", "Harvard-Oxford", "AAL", "Talairach", "Atlas"])



    for row in tadpole_data.itertuples():
        fldname_column = row[2] # row 
        text_column = row[6]  # column 6 is the TEXT column
        
        if (pd.isnull(row[6])):
            continue
        
        split_text = text_column.split()
        if (split_text[0] != "Cortical" and split_text[0] != "Surface" and split_text[0] != "Volume"):
            continue
    
        # I might be wrong about the tadpole labels being called ADNI codes
        ADNI = fldname_column.split('_')[0]

        # parse last 2 characters from ADNI code
        modality = ADNI[-2:]

        # parse description from the text column
        description = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', split_text[-1])  # credit to Markus Jarderot on stack overflow
       
        harvard_oxford_association = ""
        aal_association = ""
        talairach_association = ""
        prefered_atlas = ""
        
        for description_row in description_associations.itertuples():
            if (description_row[1] == description):
                harvard_oxford_association = description_row[2]
                aal_association = description_row[3]
                talairach_association = description_row[4]
                prefered_atlas = description_row[5]
        
        # write data from description associations file to respective modality files
        with open (target_files[modality], mode='a') as target_file:
            fsl_writer = csv.writer(target_file, delimiter=',')
            #  Columns : ["TADPOLE", "Harvard-Oxford", "AAL", "Talairach", "Atlas"]
            fsl_writer.writerow([fldname_column, harvard_oxford_association, aal_association, talairach_association, prefered_atlas])


create_atlas_map("FSL")
create_atlas_map("FSX")
