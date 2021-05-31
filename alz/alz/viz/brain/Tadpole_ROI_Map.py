import random
import pandas

from nilearn import datasets, image, plotting

""" TADPOLE Brain Visualization

Object that allows for brain plotting/viz with the TADPOLE data.

The modalities that can currenty be plotted are:
  - FSL_CV: Longitudinal FreeSurfer Volume (Cortical Parcellation)
  - FSL_SA: Longitudinal FreeSurfer Surface Area
  - FSL_SV: Longitudinal FreeSurfer Volume (White Matter Parcellation)
  - FSL_TA: Longitudinal FreeSurfer Cortical Thickness Average
  - FSL_TS: Longitudinal FreeSurfer Cortical Thickness Standard Deviation
  - FSX_CV: Cross-Sectional FreeSurfer Volume (Cortical Parcellation)
  - FSX_SA: Cross-Sectional FreeSurfer Surface Area
  - FSX_SV: Cross-Sectional FreeSurfer Volume (White Matter Parcellation)
  - FSX_TA: Cross-Sectional FreeSurfer Cortical Thickness Average
  - FSX_TS: Cross-Sectional FreeSurfer Cortical Thickness Standard Deviation
"""

class Tadpole_ROI_Map:

    roi_images = dict()
    modality_atlas_labels = None
    modality_chosen = None

    def __init__(self, modality, relative_path):
        r"""Initialize with choice of modality

        Parameters
        ----------
        modality : String,
                   Modality Options are
                   FSX_CV, FSX_SA, FSX_SV, FSX_TA, FSX_TS
                   FSL_CV, FSL_SA, FSL_SV, FSL_TA, FSL_TS
        relative_path : String,
                        relative path to alz repository
        """

        self.harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        self.aal_atlas = datasets.fetch_atlas_aal("SPM12")
        self.talairach_atlas = datasets.fetch_atlas_talairach("ba")

        self.harvard_oxford_atlas.maps = image.resample_img(self.harvard_oxford_atlas.maps)
        norm_affine = self.harvard_oxford_atlas.maps.affine
        norm_shape = tuple([2*x for x in self.harvard_oxford_atlas.maps.shape])

        self.harvard_oxford_atlas.maps = image.resample_img(self.harvard_oxford_atlas.maps,
                                                            target_affine=norm_affine,
                                                            target_shape=norm_shape)

        self.aal_atlas.maps = image.resample_img(self.aal_atlas.maps,
                                                 target_affine=norm_affine,
                                                 target_shape=norm_shape)

        self.talairach_atlas.maps = image.resample_img(self.talairach_atlas.maps,
                                                       target_affine=norm_affine,
                                                       target_shape=norm_shape)


        modalityOptions = {
            "FSX_CV" : relative_path + "alz/viz/brain/FSX_atlas_labels/FSX_CV_atlas_labels.csv",
            "FSX_SA" : relative_path + "alz/viz/brain/FSX_atlas_labels/FSX_SA_atlas_labels.csv",
            "FSX_SV" : relative_path + "alz/viz/brain/FSX_atlas_labels/FSX_SV_atlas_labels.csv",
            "FSX_TA" : relative_path + "alz/viz/brain/FSX_atlas_labels/FSX_TA_atlas_labels.csv",
            "FSX_TS" : relative_path + "alz/viz/brain/FSX_atlas_labels/FSX_TS_atlas_labels.csv",
            "FSL_CV" : relative_path + "alz/viz/brain/FSL_atlas_labels/FSL_CV_atlas_labels.csv",
            "FSL_SA" : relative_path + "alz/viz/brain/FSL_atlas_labels/FSL_SA_atlas_labels.csv",
            "FSL_SV" : relative_path + "alz/viz/brain/FSL_atlas_labels/FSL_SV_atlas_labels.csv",
            "FSL_TA" : relative_path + "alz/viz/brain/FSL_atlas_labels/FSL_TA_atlas_labels.csv",
            "FSL_TS" : relative_path + "alz/viz/brain/FSL_atlas_labels/FSL_TS_atlas_labels.csv"
        }

        self.modality_atlas_labels = pandas.read_csv(modalityOptions[modality])
        self.modality_chosen = modality


    def add_ROIs (self, ROI_weight_list):
        r"""add all ROIs from a modality with their associated weights

        Parameters
        ----------
        ROI_weight_list : array_like
                          Should be list of lists with format [tadpole label, weight]
        """

        if len(ROI_weight_list[0]) != 2:
            print("ROI weight list not formatted correctly")
        for ROI in ROI_weight_list:
            roi_label = ROI[0]
            roi_weight = ROI[1]
            roi_label_found = False

            atlas = None
            roi_idx = None
            atlas_img = None

            for index, row in self.modality_atlas_labels.iterrows():
                if row["TADPOLE"] != roi_label:
                    continue

                roi_label_found = True
                atlas = row["Atlas"]                      # set atlas equal to the atlas column of the row
                if atlas == "UNDETERMINED":
                    print(f"Atlas label for {roi_label} is undetermined")
                    break
                atlas_labels = row[atlas].split("+")      # atlas label stores the label in the column pointed to by the atlas column

                self.roi_images [roi_label] = []

                for atlas_label in atlas_labels:
                    try:
                        if atlas == "Harvard-Oxford":
                            roi_idx = self.harvard_oxford_atlas.labels.index(atlas_label)
                            atlas_img = self.harvard_oxford_atlas.maps
                        elif atlas == "AAL":
                            roi_idx = self.aal_atlas.indices[self.aal_atlas.labels.index(atlas_label)]
                            atlas_img = self.aal_atlas.maps
                        elif atlas == "Talairach":
                            roi_idx = self.talairach_atlas.labels.index(atlas_label)
                            atlas_img = self.talairach_atlas.maps
                    except ValueError:
                        print(f"{atlas_label} not found in {atlas}")
                        roi_label_found = False
                        continue

                    roi_image = image.math_img("(img == %s) * %s" % (roi_idx, roi_weight), img=atlas_img)
                    self.roi_images[roi_label].append(roi_image)

                break  # once row is found the rest of the rows don't need to be searched





    def build_map(self, smoothed):
        r"""Builds all of the images for every ROI, doesn't need to be used

        """

        #  0s out the image
        full_image_map = image.math_img("img * 0", img=self.roi_images[list(self.roi_images)[0]][0])

        # creates new map that adds every roi from the roi map to itself
        # add all roi maps together
        for roi in self.roi_images:
            for roi_image in self.roi_images[roi]:
                full_image_map = image.math_img("img1 + img2", img1=full_image_map, img2=roi_image)

        if smoothed:
            full_image_map = image.smooth_img(full_image_map, fwhm=5)


        return full_image_map



    def plot(self, plot_title="Default", smoothed=True):
        """Displays image with the image_map"""
        if plot_title is "Default":
            plot_title = self.modality_chosen
        image_map = self.build_map(smoothed)
        plotting.plot_glass_brain(image_map, black_bg=False, title=plot_title)
        plotting.show()



    def save(self, path="Default", plot_title="Default", smoothed=True):
        """Save plot to path"""
        image_map = self.build_map(smoothed)
        if path is "Default":
            path = f"out/{self.modality_chosen}_fig_trace.png"
        if plot_title is "Default":
            plot_title = self.modality_chosen

        plot = plotting.plot_glass_brain(image_map,
                                         black_bg=False,
                                         title=plot_title,
                                         output_file=path,
                                         colorbar=True)


    def get_modality_labels (self):

        all_labels = []

        for index,row in self.modality_atlas_labels.iterrows():

            all_labels.append(row['TADPOLE'])
        return all_labels
