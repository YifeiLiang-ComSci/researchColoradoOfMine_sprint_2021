import numpy as np
import utilsforminds.helpers as helpers

def push_arr_to_range(nparr, vmin = None, vmax = None):
    if vmin is not None and vmax is not None:
        return np.where(nparr > vmax, vmax, np.where(nparr < vmin, vmin, nparr))
    elif vmin is not None and vmax is None:
        return np.where(nparr < vmin, vmin, nparr)
    elif vmin is None and vmax is not None:
        return np.where(nparr > vmax, vmax, nparr)
    else:
        # print("WARNING: vmin and vmax both are None, so nparr is not pushed to range.")
        return np.copy(nparr)

def get_value_of_proportional_rank(amount_arr, proportional_rank, mask_arr = None):
    """Get the value corresponding to the given rank.
    
    The rank is counted from the smallest value, so the proportional_rank 0.7 of 1~10 is 7.

    Examples
    --------
    amount_arr = np.array([[1., 2., 3.], [4., 5., 6.]])\n
    mask_arr = np.array([[1., 1., 0.], [1., 1., 1.]])\n
    print(get_value_of_proportional_rank(amount_arr = amount_arr, mask_arr = mask_arr, proportional_rank = 0.7))
        : 5.0
    print(get_value_of_proportional_rank(amount_arr = amount_arr, mask_arr = mask_arr, proportional_rank = 0.3))
        : 2.0
    """

    assert(0. <= proportional_rank and proportional_rank <= 1.)
    mask_arr_clean = np.where(mask_arr >= 1., 1., 0.) if mask_arr is not None else np.ones(amount_arr.shape)
    numEntriesImputed = np.count_nonzero(mask_arr_clean)
    if numEntriesImputed > 0:
        rank = int(numEntriesImputed * proportional_rank) ## values higher than the rank from the 'smallest' value will survive, in other words maskKeepThreshold * 100% will die.
        rankValue = np.partition(amount_arr[np.nonzero(mask_arr_clean)], rank)[rank]
        return rankValue
    else:
        return 1e-16

def slice_array_to_given_shape(arr, shape, origin = "center"):
    """Get the sub-array with given shape.
    
    Parameters
    ----------
    origin : tuple, str
        If tuple, it is (x_begin, y_begin, z_begin), else if 'center', it will slice the center of array.
    
    Examples
    --------
    test_arr = np.zeros((100, 100, 100))
    print(slice_array_to_given_shape(arr= test_arr, shape= [60, 200, 70], origin = "center").shape)
        >>> (60, 100, 70)
    """

    slice_dict = {}
    for axis in range(3):
        if arr.shape[axis] > shape[axis]:
            if origin == "center":
                slice_dict[axis] = [(arr.shape[axis] - shape[axis]) // 2, (arr.shape[axis] - shape[axis]) // 2 + shape[axis]]
            else:
                assert(arr.shape[axis] >= origin[axis] + shape[axis])
                slice_dict[axis] = [origin[axis], origin[axis] + shape[axis]]
    return helpers.getSlicesV2(npArr= arr, dimIdxDict= slice_dict)

def mask_prob(shape, p):
    """
        Generates a random mask array of 1's and 0's of the specified shape and the given probability. Returns as type np.float32. The larger p results the denser (more 1's) mask.
    """

    A = np.random.uniform(0., 1., size = shape) # Generate a random standard uniform distribution of the specified shape
    B = A < p   # Where A < p, a True (1) value is recorded in B, if A > p then a False (0) vlaue is recorded in B
    return (1. * B).astype(np.float32)  # Return a mask array of 0s and 1s

def inverse_one_hot_encode(arr, return_encode_list = False):
    """

    Parameters
    ----------
    arr : Numpy array
        One-hot encoded array. The dimension should be larger than 2 and arr.shape[0] is interpreted as the number of samples.

    Examples
    --------
    test_arr = np.array([[0., 1.], [0., 1.], [1., 0.], [0., 1.]])
    print(inverse_one_hot_encode(test_arr))
    >>> [0 0 1 0]

    test_arr = np.array([[0., 1.], [0., 1.], [1., 0.], [0., 1.]])
    print(inverse_one_hot_encode(test_arr, return_encode_list = True))
    >>> (array([0, 0, 1, 0]), [array([0., 1.]), array([1., 0.])])
    """

    unique_labels = np.unique(arr, axis= 0)
    arr_single_label = []
    for i in range(arr.shape[0]):
        for j in range(unique_labels.shape[0]):
            if np.array_equal(arr[i], unique_labels[j]):
                arr_single_label.append(j)
                break
    arr_single_label = np.array(arr_single_label)
    assert(arr_single_label.shape[0] == arr.shape[0])
    if not return_encode_list:
        return arr_single_label
    else:
        encode_dict = [unique_labels[j] for j in range(unique_labels.shape[0])]
        return arr_single_label, encode_dict

if __name__ == "__main__":
    pass
    counts_dict = {(lambda x: True if x > 2 else False): 0, (lambda x: True if x <= 2 else False): 0}
    for i in range(10):
        for key in counts_dict.keys():
            if key(i):
                counts_dict[key] += 1
    
    print(counts_dict)
    