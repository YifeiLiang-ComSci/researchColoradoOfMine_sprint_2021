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

#choose a modality
modalities={'SNP':('snp_','genetic/'),'ApoE':('apoe_','genetic/'),"CSF":('csf_','csf/'),
            "Cognitive":('cog_','cognitive_tests/'),"MRI UCSF":('mri_ucsf_','mri/'),
            "MRI Longitudinal FreeSurfer":('mri_lfs_','mri/'),"MRI CrossSectional FreeSurfer":('mri_xfs_','mri/'),
            "DTI":('dti_','mri/'),"PET Averages":('pet_avg_','pet/'),"PET Banner Institute":('pet_bai_','pet/'),
            "PET AV45":('pet_av45_','pet/'),"PET AV1451":('pet_av1451_','pet/'),
            "Demographics":('dem_','demographics/'),"Diagnosis":('dx_','diagnosis/'),"All data":('','')}
choice='CSF'
folder=modalities[choice][1]
filename=modalities[choice][0]

## Load data
data=np.load('../data/tadpole/tadpole_npy/'+folder+filename+'data.npy', allow_pickle=True)
labels=np.load('../data/tadpole/tadpole_npy/'+folder+filename+'labels.npy', allow_pickle=True)

# +
## see what data you have

## these are same for all full datasets
# print('Patients available: '+' '.join([str(i) for i in list(labels[0])]))
# print('Time points available: '+' '.join([str(i) for i in list(labels[1])]))

## featues
print('Features available: '+', '.join([str(i) for i in list(labels[2])]))

# +
## accessing data
patient=31
time='bl'
feature='TAU_UPENNBIOMK9_04_19_17'

## get index
patient_index=list(labels[0]).index(patient)
time_index=list(labels[1]).index(time)
feature_index=list(labels[2]).index(feature)
# -

## at a certain patient, time point, feature
#note that these values are strings not floats. this is easy to convert to a float.
data[patient_index,time_index,feature_index]

## all features at a time point and patient
data[patient_index,time_index,:]

## compressed data is only time points and patients with data
## Load data
compressed_data=np.load('../data/tadpole/tadpole_npy/'+folder+'compressed_'+filename+'data.npy', allow_pickle=True)
compressed_labels=np.load('../data/tadpole/tadpole_npy/'+folder+'compressed_'+filename+'labels.npy', allow_pickle=True)

# +
## see what data you have

## these are same for all full datasets
print('Patients available: '+' '.join([str(i) for i in list(compressed_labels[0])]))
print('Time points available: '+' '.join([str(i) for i in list(compressed_labels[1])]))

## featues
print('Features available: '+', '.join([str(i) for i in list(compressed_labels[2])]))

# +
#plot how many patients have a certian modality/a certian part of a modality
#note: this assumes if you have any data for a patient,time you have all data

times=list(labels[1]) #times
count=[0]*len(times) #how much data at each time, initializes to zero

for t in range(len(times)):
    #if some data at that time
    if(times[t] in list(compressed_labels[1])):
        index=list(compressed_labels[1]).index(times[t]) #get index of time in compressed version
        for i in [g for h in compressed_data[:,index,:] for g in h]: #concatanates into one list
            #test if not nan
            if i == i:
                count[t]+=1 #add one everytime a pateint has a feature at that time
        count[t]=count[t]/len(list(labels[2])) #divide by number of features
        
plt.bar(np.arange(len(list(labels[1]))),count,align='center')
plt.xlabel('time')
plt.ylabel('count')
plt.title(choice)
plt.xticks(np.arange(len(times)),times,rotation='vertical')
plt.show()

# +
## plot feature over time for patient

## accessing data
patient=31
feature='TAU_UPENNBIOMK9_04_19_17'

## get index
patient_index=list(labels[0]).index(patient)
feature_index=list(labels[2]).index(feature)

# get y values converting to float and x values
y=np.array([float(i) for i in data[patient_index,:,feature_index]])
x=np.arange(len(labels[1]))

#create lists without nan values
mask = [i==i for i in list(y)]
x2=x[mask]
y2=y[mask]

# plot lines where consecutive values
plt.plot(x,y,'k-')
# plot dotted lines where non consecutive values
plt.plot(x2,y2,'k--')
# plot dots on measured values
plt.plot(x,y,'k.')

plt.xlabel('time')
plt.ylabel('test score')
plt.title(feature+' for patient '+str(patient))
plt.xticks(np.arange(len(labels[1])),labels[1],rotation='vertical')
plt.show()
# -


