import pandas as pd

def get_amount_counter_min_max_from_excel(path_to_xlsx, shape = (80, 80, 80)):
    df = pd.read_excel(path_to_xlsx)
    x = np.array(df["East.m"])
    y = np.array(df["North.m"])
    z = np.array(df["Sample.Elevm"])
    min_max_tuple = [[np.min(x), np.min(y), np.min(z)], [np.max(x), np.max(y), np.max(z)]]
    grid = [(min_max_tuple[1][0] - min_max_tuple[0][0]) / shape[0], (min_max_tuple[1][1] - min_max_tuple[0][1]) / shape[1], (min_max_tuple[1][2] - min_max_tuple[0][2]) / shape[2]]
    amount = np.array(df["N_160"])

    i = np.round((x - min_max_tuple[0][0]) / grid[0]).astype(np.int)
    j = np.round((y - min_max_tuple[0][1]) / grid[1]).astype(np.int)
    k = np.round((z - min_max_tuple[0][2]) / grid[2]).astype(np.int)

    i = np.where(i >= shape[0], shape[0] - 1, i)
    j = np.where(j >= shape[1], shape[1] - 1, j)
    k = np.where(k >= shape[2], shape[2] - 1, k)

    amount_arr = np.zeros(shape)
    counter_arr = np.zeros(shape)
    amount_arr[i, j, k] = amount
    counter_arr[i, j, k] = 1.
    return amount_arr, counter_arr, min_max_tuple

def getDataDensity(npArr):
    """Get the proportion of non-zero elements.

    Works for any dimension
    
    """

    noEntries = 1
    for length in npArr.shape:
        noEntries = noEntries * length
    return 1. - np.count_nonzero(npArr == 0.)/float(noEntries)