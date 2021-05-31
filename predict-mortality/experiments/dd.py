import pickle
dataset_kind = "chestxray"  # covid, challenge, alz, toy, chestxray
dataset_path = f"./datasets/{dataset_kind}_saasdasde.obj"
dataset_obj = None
with open(dataset_path, "wb") as dataset_file:
    pickle.dump(dataset_obj, dataset_file)
