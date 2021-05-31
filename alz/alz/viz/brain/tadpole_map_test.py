import random
from Tadpole_ROI_Map import Tadpole_ROI_Map


roi_map = Tadpole_ROI_Map("FSX_CV")

weights = []
for label in roi_map.get_modality_labels():
    weight = random.randint(0, 100)
    weights.append([label, weight])


roi_map.add_ROIs(weights)
roi_map.plot()
roi_map.save()

