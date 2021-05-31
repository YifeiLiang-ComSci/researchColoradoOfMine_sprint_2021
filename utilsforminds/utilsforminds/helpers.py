import numpy as np
from random import random
import os
import sys
import math
import numbers
import pickle
import moviepy.editor as mpy
from copy import deepcopy
import pandas as pd

# from sklearn.grid_search import ParameterGrid # moved in recent version of sklearn
from sklearn.model_selection import ParameterGrid
import inspect

import shutil
from PIL import Image, ImageSequence
import tensorflow as tf
import utilsforminds.visualization as visualization

from datetime import timezone, datetime

axisMap = {0: 'x', 1: 'y', 2: 'z', 'x': 0, 'y': 1, 'z': 2}

def getExecPath():
    '''
        ref: https://stackoverflow.com/questions/606561/how-to-get-filename-of-the-main-module-in-python
    '''
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__)
    except:
        sFile = sys.executable
    return os.path.dirname(sFile)

def getNewDirectoryName(parentDir, newDir, root_dir = None):
    '''
        To get new directory name to save results while avoiding duplication
    '''

    if root_dir is None:
        root_dir = getExecPath()
    if parentDir[0] != '/':
        parentDir = '/' + parentDir
    if parentDir[-1] != '/':
        parentDir = parentDir + '/'

    assert(root_dir + parentDir)

    duplicatedNameNum = 0
    while(os.path.isdir(root_dir + parentDir + newDir + str(duplicatedNameNum))):
        duplicatedNameNum = duplicatedNameNum + 1
    newDir = newDir + str(duplicatedNameNum)

    return newDir

# def getPlaneAverageData(npArr, normalDirection):  # https://stackoverflow.com/questions/37142135/sum-numpy-ndarray-with-3d-array-along-a-given-axis-1
#     assert(normalDirection in ('x', 'y', 'z'))
#     if normalDirection == 'x':
#         return np.average(npArr, axis=0)
#     elif normalDirection == 'y':
#         return np.average(npArr, axis=1)
#     else:
#         return np.average(npArr, axis=2)

def getDataDensity(npArr):
    """Get the proportion of non-zero elements.

    Works for any dimension
    
    """

    noEntries = 1
    for length in npArr.shape:
        noEntries = noEntries * length
    return 1. - np.count_nonzero(npArr == 0.)/float(noEntries)

# def getMostObsIdx(npArr3D, axis):
#     assert(axis in ('x', 'y', 'z'))
#     maxCount = -1
#     maxIdx = 0
#     if axis == 'x':
#         for i in range(npArr3D.shape[0]):
#             if maxCount <= np.count_nonzero(npArr3D[i, :, :]):
#                 maxIdx = i
#                 maxCount = np.count_nonzero(npArr3D[i, :, :])
#         return maxIdx
#     elif axis == 'y':
#         for i in range(npArr3D.shape[1]):
#             if maxCount <= np.count_nonzero(npArr3D[:, i, :]):
#                 maxIdx = i
#                 maxCount = np.count_nonzero(npArr3D[:, i, :])
#         return maxIdx
#     else: # axis == 'z':
#         for i in range(npArr3D.shape[2]):
#             if maxCount <= np.count_nonzero(npArr3D[:, :, i]):
#                 maxIdx = i
#                 maxCount = np.count_nonzero(npArr3D[:, :, i])
#         return maxIdx


def getNMAE(ground_truth_amount_arr, recovered_arr, ground_truth_counter_arr):
    """Get Normalized Mean Absolute Error.

    Works for any dimension.
    
    """

    assert(ground_truth_amount_arr.shape == recovered_arr.shape and recovered_arr.shape == ground_truth_counter_arr.shape)
    rMax = ground_truth_amount_arr.max()
    rMin = ground_truth_amount_arr.min()
    groundTruthCounterTensor01 = np.where(ground_truth_counter_arr >= 1., 1., 0.)
    if np.count_nonzero(groundTruthCounterTensor01) == 0. or rMax == rMin:
        return 0.
    return np.sum(np.absolute(ground_truth_amount_arr - recovered_arr) * groundTruthCounterTensor01) / (np.sum(groundTruthCounterTensor01) * (rMax - rMin))

def getSparsedDataCounterRandom(npArrData, npArrCounter, samplingProb):
    """Get regularly sampled mineral amount array and counter array.

    Parameters
    ----------
    npArrData : Numpy array
        Array of mineral amounts.
    npArrCounter : Numpy array
        Array of the number of observations.
    samplingProb : float
        Probability to KEEP the observations. So large samplingProb gives more dense sampled array.
    
    Returns
    -------
    npArrData * boolArr : Numpy array
        Sampled mineral amount array.
    npArrCounter * boolArr : Numpy array
        Sample counter array.
    """

    assert(samplingProb <= 1. and samplingProb >= 0.)
    assert(npArrCounter.shape == npArrData.shape)
    
    shape_ = npArrData.shape
    randomArr = np.random.uniform(0., 1., size = shape_)
    boolArr = randomArr < samplingProb
    return npArrData * boolArr, npArrCounter * boolArr

def getSparsedDataCounterCuvicHole(npArrData, npArrCounter, lstOfProportions):
    """Get irregularly(cubic shape) sampled mineral amount array and counter array.

    Works only for 3D or 4D array.
    Parameters
    ----------
    npArrData : Numpy array
        Array of mineral amounts.
    npArrCounter : Numpy array
        Array of the number of observations.
    lstOfProportions : list of floats
        Volume proportions of REMOVAL cubes. 
        For example, [0.1, 0.1, 0.1] will create three blanks with 10% volume proportions.
        So large proportions give more sparse sampled array.
    
    Returns
    -------
    npArrData * samplingArr : Numpy array
        Sampled mineral amount array.
    npArrCounter * samplingArr : Numpy array
        Sample counter array.
    """

    # assert(sum(lstOfProportions) <= 1. and sum(lstOfProportions) >= 0.)
    for proportion in lstOfProportions:
        assert(proportion > 0. and proportion <= 1.)
    assert((len(npArrData.shape) == 3 or len(npArrData.shape) == 4) and npArrData.shape == npArrCounter.shape)

    shape_ = npArrData.shape
    samplingArr = np.ones(shape_)

    for proportion in lstOfProportions:
        proportion = proportion ** (1/3)
        cubeSize = (math.floor(shape_[0] * proportion), math.floor(shape_[1] * proportion), math.floor(shape_[2] * proportion))
        startingIdc = []
        for axis in range(3):
            startingIdc.append(random.randint(0, shape_[axis] - cubeSize[axis]))
        if len(npArrData.shape) == 3:
            samplingArr[startingIdc[0]: startingIdc[0] + cubeSize[0], startingIdc[1]: startingIdc[1] + cubeSize[1], startingIdc[2]: startingIdc[2] + cubeSize[2]] = 0.
        else:
            samplingArr[startingIdc[0]: startingIdc[0] + cubeSize[0], startingIdc[1]: startingIdc[1] + cubeSize[1], startingIdc[2]: startingIdc[2] + cubeSize[2], :] = 0.
    print(f'Proportion of sampled entries: {np.count_nonzero(npArrCounter * samplingArr)/np.count_nonzero(npArrCounter)}')

    return npArrData * samplingArr, npArrCounter * samplingArr

# def getScaledMinMax(npArr2D):
#     """
#         Deprecated
#     """
#     shape = npArr2D.shape
#     assert(len(shape) == 2)
#     max = npArr2D.max()
#     min = npArr2D.min()
#     scaled = np.zeros(shape)

#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             scaled[i, j] = (npArr2D[i, j] - min)/(max - min)
    
#     return scaled, min, max

def min_max_scale(arr, vmin = None, vmax = None, arr_min = None, arr_max = None):
    """Apply min-max scaling. This changes close to the given range, e.g. [vmin, vmax] but not exactly if arr_min and arr_max is given differently from min and max of arr. If you changes vmin <-> arr_min, vmax <-> arr_max, then this becomes reverse scaling to original exactly same arr.
    
    The range changes from [arr_min, arr_max],
    to [vmin, vmax] if vmin and vmax are both given, 
    to [vmin, vmin + 1] if only vmin is given, 
    to [vmax - 1, vmax] if only vmax is given,
    to [0, 1] if vmin and vmax are both not given.
    """

    max_ = np.max(arr) if arr_max is None else arr_max
    min_ = np.min(arr) if arr_min is None else arr_min
    if max_ == min_:
        arr_01_scaled = np.ones(arr.shape) * 0.5
    else:
        arr_01_scaled = (arr - min_) / (max_ - min_) ## [0, 1] scaled.
    if vmax is not None and vmin is not None:
        assert(vmax >= vmin)
        return arr_01_scaled * (vmax - vmin) + vmin
    elif vmax is not None and vmin is None:
        return arr_01_scaled + (vmax - 1.)
    elif vmax is None and vmin is not None:
        return arr_01_scaled + vmin
    else:
        return arr_01_scaled
    

def reverseMinMaxScale(arr, min_, max_, onlyPositive = False):
    raise Exception("Deprecated Function, use utilsforminds.helpers.min_max_scale instead.")
    if max_ > min_:
        reversedScaled = arr * (max_ - min_) + min_
        if onlyPositive:
            return np.where(reversedScaled >= 0., reversedScaled, 0.)
        else:
            return reversedScaled
    elif max_ == min_:
        print(f'WARNING: min_: {min_} == max_: {max_}')
        return np.where(arr == min_, 0.5, 0.)
    else:
        raise Exception(f'max: {max_} is smaller than min: {min_}')

def is_small_container(container, length_limit = 20):
    if (type(container) == type({}) or type(container) == type([]) or type(container) == type(tuple([3, 3]))) and len(container) < length_limit:
        if type(container) == type({}):
            for key, value in container.items():
                if not (isinstance(key, (numbers.Number, type('a'), type(True), type(None))) and isinstance(value, (numbers.Number, type('a'), type(True), type(None)))):
                    return False
            return True
        if type(container) == type([]) or type(container) == type(tuple([3, 3])):
            for item in container:
                if not (isinstance(item, (numbers.Number, type('a'), type(True), type(None)))):
                    return False
            return True
    else:
        False

def load_csv_columns_into_list(path_to_csv: str):
    result_dict = {}
    with open(path_to_csv, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter = ',')
        headers = csv_reader.fieldnames
        for header in headers:
            result_dict[header] = []
        for line in csv_reader:
            for header in line.keys():
                result_dict[header].append(line[header])
    del result_dict['']
    return result_dict

def paramDictToStr(param_dict):
    if isinstance(param_dict, (numbers.Number, type("a"))):
        return f"{param_dict}\n"
    elif type(param_dict) != type({}):
        return "None\n"
    else:
        result = ""
        for key in param_dict.keys():
            if (isinstance(key, (numbers.Number, type('a'), type(True), type(None))) or is_small_container(key)) and (isinstance(param_dict[key], (numbers.Number, type('a'), type(True), type(None))) or is_small_container(param_dict[key])):
                result = result + str(key) + " : " + str(param_dict[key]) + "\n"
        return result

def gridSearch(function, params_grid):
    """Deprecated, use containers.get_list_of_grids."""
    duplicatedNameNum = 0
    while(os.path.isfile(os.path.dirname(__file__) + '/gridSearchResults/' + function.__name__ + '_' + str(duplicatedNameNum) + ".txt")):
        duplicatedNameNum = duplicatedNameNum + 1
    txtFile = open(os.path.dirname(__file__) + '/gridSearchResults/' + function.__name__ + '_' + str(duplicatedNameNum) + ".txt", "w")

    grid = ParameterGrid(params_grid)
    bestParams = {}
    bestScore = -1. * sys.float_info.max
    for params in grid:
        largerBetterScore = function(**params)
        if largerBetterScore >= bestScore:
            bestScore = largerBetterScore
            bestParams = params
            print(str(bestParams) + "\n\t" + str(bestScore) + "\n")
        txtFile.write(str(params) + "\n\t" + str(largerBetterScore) + "\n")
    
    assert(bestScore >= 0.)
    txtFile.write("\n\n--best result--\n" + str(bestParams) + "\n\t" + bestScore)
    txtFile.close()

# def makeTestArr(shape):
#     assert(len(shape) == 4)
#     result = []
#     for i in range(shape[0]):
#         result.append([])
#         for j in range(shape[1]):
#             result[i].append([])
#             for k in range(shape[2]):
#                 result[i][j].append([])
#                 for l in range(shape[3]):
#                     result[i][j][k].append(str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l))
#     return np.array(result)

def splitRange(isNoPartitions, number, start, end):
    """
    
    Examples
    --------
    helpers.splitRange(True, 3, 2, 15) : 
        [(2, 5), (7, 10), (12, 15)]
    helpers.splitRange(False, 3, 2, 15) :
        [(2, 3), (5, 6), (8, 9), (11, 12), (14, 15)]
    helpers.splitRange(False, 6, 2, 15) : 
        [(2, 6), (8, 12), (14, 15)]
    helpers.splitRange(True, 6, 2, 15) : 
        [(2, 3), (5, 6), (8, 9), (11, 12), (14, 15)]
    """

    assert(number > 0 and end > start and end > 0 and start >= 0)
    if not isNoPartitions:
        assert(number < end - start)
    if isNoPartitions:
        partSize = math.ceil((end - start) / number)
    else:
        partSize = number
    result = []
    startIdx = start
    endIdx = partSize
    while(True):
        if(startIdx + partSize >= end):
            result.append((startIdx, end))
            break
        result.append((startIdx, endIdx))
        startIdx = startIdx + partSize
        endIdx = endIdx + partSize
    return result

def splitList(isNoPartitions, number, lst):
    """
    
    Examples
    --------
    helpers.splitList(True, 5, list(range(31))) : 
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27], [28, 29, 30]]
    helpers.splitList(True, 5, list(range(30))) :
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29]]
    helpers.splitList(True, 5, list(range(29))) : 
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28]]
    helpers.splitList(False, 5, list(range(31))) : 
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30]]
    """

    assert(number > 0 and len(lst) > 0)
    if not isNoPartitions:
        number < len(lst)
    ranges = splitRange(isNoPartitions, number, 0, len(lst))
    result = []
    for rng in ranges:
        result.append(lst[rng[0]:rng[1]])
    return result

def fromDictToDirName(dict, limitLength = 150):
    dirName = ''
    for key in dict.keys():
        if (isinstance(key, numbers.Number) or isinstance(key, str)) and (isinstance(dict[key], numbers.Number) or isinstance(dict[key], str)):
            dirName = dirName + str(key).replace('.', 'p') + '-' +str(dict[key]).replace('.', 'p') + '_'
    assert(dirName != '')
    dirName = dirName[:-1]
    if len(dirName) > limitLength:
        return dirName[:limitLength]
    else:
        return dirName

def getDirSize(pathStart): # https://stackoverflow.com/questions/20267220/how-to-check-size-of-the-files-in-a-directory-with-python
    bytes = 0
    for f in os.listdir(pathStart):
        path = os.path.join(pathStart, f)
        if os.path.isfile(path):
            bytes = bytes + os.path.getsize(path)
    return bytes / (1000 * 1000 * 1000) # return in GB unit

def getAvgLstFrom4DArr(dataArr, counterArr):
    '''
        Get average for each feature on entries whose counterArr > 1.
    '''

    assert(len(dataArr.shape) == 4 and dataArr.shape == counterArr.shape)
    avgLst = []
    for feature in range(dataArr.shape[3]):
        mask01 = np.where(counterArr[:, :, :, feature] >= 1., 1., 0.)
        avgLst.append(np.sum(dataArr[:, :, :, feature] * mask01) / (np.count_nonzero(mask01) + 1e-8))
    return avgLst
    
def getRMSE(groundTruthValueArr, targetValueArr, groundTruthMaskArr):
    """Get Root Mean Squared Error

    Works for any dimension.
    
    """
    groundTruthMaskArr01 = np.where(groundTruthMaskArr >= 1., 1., 0.)
    return np.sum(groundTruthMaskArr01 * (groundTruthValueArr - targetValueArr) ** 2.) ** 0.5 / np.sum(groundTruthMaskArr01)

# def reverseLst(lst):
#     resultLst = []
#     for i in range(len(lst)):
#         resultLst.append(lst[len(lst) - 1 - i])
#     return resultLst

def compressNparrLst(lst):
    assert(type(lst) == type([]) and len(lst) > 0)
    shape_ = lst[0].shape
    counterNparr = np.zeros(shape_)
    valueNparr = np.zeros(shape_)
    with np.errstate(divide='ignore',invalid='ignore'):
        for idx in range(len(lst)):
            assert(lst[idx].shape == shape_)
            counterNparr = np.where(lst[idx] > 1e-8, counterNparr + 1, counterNparr)
            valueNparr = np.where(counterNparr > 1e-8, np.where(lst[idx] > 1e-8, (valueNparr * (counterNparr - 1) + lst[idx]) / counterNparr, valueNparr), valueNparr)
    
    return valueNparr

# def getSlices(npArr, dimIdxDict):
#     """Deprecated. Use getSlicesV2 instead.

#         example: getSlices(arr, {2: 3, 0: 1}) == arr[1, :, 3]
#         getSlices(arr, {2: [1, 3], 0:1}) == arr[1, :, 1:3]
#     """
#     dims = []
#     idxs = []
#     for dim in dimIdxDict.keys():
#         dims.append(dim)
#         if type(dimIdxDict[dim]) == type([]) or type(dimIdxDict[dim]) == type(()):
#             assert(len(dimIdxDict[dim]) == 2)
#             assert(npArr.shape[dim] >= dimIdxDict[dim][1])
#             idxs.append(list(range(dimIdxDict[dim][0], dimIdxDict[dim][1])))
#         else:
#             assert(npArr.shape[dim] > dimIdxDict[dim])
#             idxs.append(dimIdxDict[dim])
    
#     movedDims = list(range(len(dims)))
#     return np.moveaxis(npArr, dims, movedDims)[tuple(idxs)]

def getSlicesV2(npArr, dimIdxDict):
    """Get slices given tuple.

    ref: https://stackoverflow.com/questions/39474396/building-a-tuple-containing-colons-to-be-used-to-index-a-numpy-array
    
    Examples
    --------
        getSlicesV2(arr, {2: 3, 0: 1}) == arr[1, :, 3]
        getSlicesV2(arr, {2: [1, 3], 0:1}) == arr[1, :, 1:3]

    """

    indices = []
    for dim in range(len(npArr.shape)):
        if dim not in dimIdxDict.keys():
            indices.append(slice(None))
        elif type(dimIdxDict[dim]) == type([]) or type(dimIdxDict[dim]) == type(()):
            if len(dimIdxDict[dim]) == 2:
                indices.append(slice(dimIdxDict[dim][0], dimIdxDict[dim][1]))
            elif len(dimIdxDict[dim]) == 2:
                indices.append(slice(dimIdxDict[dim][0], dimIdxDict[dim][1], dimIdxDict[dim][2]))
            else:
                raise Exception(f'wrong length of slice:{len(dimIdxDict[dim])}, this should be 2 or 3')
        else:
            indices.append(dimIdxDict[dim])
    return npArr[tuple(indices)]

def get_slices_with_idc_rec(arr, dim_idc_dict):
    """ Helper recursive function for get_slices_with_idc().

    Don't use this directly.
    
    """

    if len(dim_idc_dict) == 0:
        return arr
    elif len(dim_idc_dict) == 1:
        single_key = next(iter(dim_idc_dict))
        return np.take(a = arr, indices = dim_idc_dict[single_key], axis = single_key)
    else:
        single_key = next(iter(dim_idc_dict))
        single_item = dim_idc_dict[single_key]
        del dim_idc_dict[single_key]
        return get_slices_with_idc_rec(np.take(arr, indices = single_item, axis = single_key), dim_idc_dict)

def get_slices_with_idc(arr, dim_idc_dict):
    """ Slicing or Indexing Numpy array with given dlictionary {axis: indices}.

    Examples
    --------
    get_slices_with_idc(arr, {0: [2, 3], 3: [4, 1, 0]}) == arr[(2, 3),:,:,(4, 1, 0)]\n
    arr.shape == (2, 4, 3) => get_slices_with_idc(arr, 1: [2]).shape == (2, 1, 3), not (2, 3).
    
    """

    dim_idc_dict_copied = deepcopy(dim_idc_dict)
    return get_slices_with_idc_rec(arr, dim_idc_dict_copied)

def is_number_repl_isdigit(s):
    """Returns True is string is a digit. 

    ref : https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    
    Parameters
    ----------
    s : str
        Input string.
    returns
    -------
    : bool
        Whether s is digit or not.
    """

    return s.replace('.','',1).isdigit()

# def deleteKeysFromDct(dct, keys):
#     """Delete elements from dictionary dct whose key is in keys
    
#     Returns
#     -------
#     deletedDct : dict
#     """

#     deletedDct = dict(dct)
#     for key in keys:
#         del deletedDct[key]
#     return deletedDct

# def popFromLstWithIdc(lst, idc):
#     """Pop multiple elements from lst
    
#     Parameters
#     ----------
#     lst : list
#         list to be poped
#     idc : iterable
#         indices
    
#     Returns
#     -------
#     poped : list
#         list of poped elements

#     Side effects
#     ------------
#     lst will be changed
#     """

#     idc = sorted(idc, reverse = True)
#     poped = []
#     for idx in idc:
#         poped.append(lst.pop(idx))
#     return poped

def getNMAE_window_rec(groundTruthTensor, groundTruthCounterTensor, recoveredTensor, window, stride = None):
    """Recursive function to get window-wise absolute error, number of observations, max, and min
    
    This is designed to be used in getNMAE_window function

    Parameters
    ----------
    groundTruthTensor : Numpy array
        array of ground truth values
    groundTruthCounterTensor : Numpy array
        array of the number of observations
    recoveredTensor : Numpy array
        array of the imputed values
    window : Numpy array
        window which will be applied on the values of data as convolution
    stride : list or tuple
        step sizes for each axis

    Returns
    -------
    sumNumObsv : float
        window-wise absolute error
    sumNumObsv : float
        window-wise number of observations
    rMax : float
        window-wise maximum value
    rMin : float
        window-wise minimum value
    """

    shape_ = groundTruthTensor.shape
    assert(len(groundTruthCounterTensor.shape) == len(shape_))
    assert(len(recoveredTensor.shape) == len(shape_))
    assert(len(window.shape) == len(shape_))
    if stride is None:
        stride = []
        for axis in range(len(shape_)):
            stride.append(1)
    assert(len(stride) == len(shape_))

    rMax = - sys.float_info.max
    rMin = sys.float_info.min

    axisToMove = 0
    for axis in range(len(shape_) - 1):
        if shape_[axisToMove] != window.shape[axisToMove]:
            break
        else:
            axisToMove = axisToMove + 1
    if axisToMove == len(shape_) - 1: # Base case: last dim
        sumNMAE = 0.
        sumNumObsv = 0.
        position = 0
        while(position + window.shape[axisToMove] <= shape_[axisToMove]):
            groundTruthConv = np.sum(window * getSlicesV2(groundTruthTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}) * getSlicesV2(groundTruthCounterTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}))
            recoveredConv = np.sum(window * getSlicesV2(recoveredTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}) * getSlicesV2(groundTruthCounterTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}))
            numObsvConv = np.sum(window * getSlicesV2(groundTruthCounterTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}))
            sumNMAE = sumNMAE + abs(groundTruthConv - recoveredConv)
            sumNumObsv = sumNumObsv + numObsvConv
            if rMax < groundTruthConv:
                rMax = groundTruthConv
            if rMin > groundTruthConv:
                rMin = groundTruthConv
            position = position + stride[axisToMove]
        return sumNMAE, sumNumObsv, rMax, rMin
    else: # Recursive case
        sumNMAE = 0.
        sumNumObsv = 0.
        position = 0
        while(position + window.shape[axisToMove] <= shape_[axisToMove]):            
            partialNMAE, partialNumObsv, partial_rMax, partial_rMin = getNMAE_window_rec(getSlicesV2(groundTruthTensor, {axisToMove:[position, position + window.shape[axisToMove]]}), 
            getSlicesV2(groundTruthCounterTensor, {axisToMove:[position, position + window.shape[axisToMove]]}), 
            getSlicesV2(recoveredTensor, {axisToMove:[position, position + window.shape[axisToMove]]}), window, stride)

            sumNMAE = sumNMAE + partialNMAE
            sumNumObsv = sumNumObsv + partialNumObsv

            if rMax < partial_rMax:
                rMax = partial_rMax
            if rMin > partial_rMin:
                rMin = partial_rMin

            position = position + stride[axisToMove]
        return sumNMAE, sumNumObsv, rMax, rMin

def getNMAE_window(groundTruthTensor, recoveredTensor, groundTruthCounterTensor, window, stride = None):
    """Get window-wise NMAE.

    Works on any dimensional Numpy array.
    
    Parameters
    ----------
    groundTruthTensor : Numpy array
        array of ground truth values
    groundTruthCounterTensor : Numpy array
        array of the number of observations
    recoveredTensor : Numpy array
        array of the imputed values
    window : Numpy array
        window which will be applied on the values of data as convolution
    stride : list or tuple
        step sizes for each axis
    
    Returns
    -------
    NMAE : float
        window-wise NMAE
    """

    if type(window) == type(()) or type(window) == type([]):
        window_ = np.array(window)
    else:
        window_ = window 
    shape_ = groundTruthTensor.shape
    assert(len(groundTruthCounterTensor.shape) == len(shape_))
    assert(len(recoveredTensor.shape) == len(shape_))
    assert(len(window_.shape) == len(shape_))
    if stride is None:
        stride = []
        for axis in range(len(shape_)):
            stride.append(1)
    assert(len(stride) == len(shape_))

    sumNMAE, sumNumObsv, rMax, rMin = getNMAE_window_rec(groundTruthTensor, groundTruthCounterTensor, recoveredTensor, window_, stride)
    return sumNMAE / ((sumNumObsv + 1e-8) * max((rMax - rMin), 1e-8))

# test_groundTruthTensor = np.random.rand(300, 300, 300, 9) * 10
# test_groundTruthCounterTensor = np.random.rand(300, 300, 300, 9) * 2
# test_recoveredTensor = np.random.rand(300, 300, 300, 9) * 10
# window = np.ones((2, 2, 2, 9))
# stride = [1, 1, 1, 1]

# print(getNMAE_window(test_groundTruthTensor, test_groundTruthCounterTensor, test_recoveredTensor, window, stride))

# ref: https://stackoverflow.com/questions/2651874/embed-bash-in-python
def run_script(script, stdin=None):
    """Returns (stdout, stderr), raises error on non-zero return code"""
    import subprocess
    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
    # cause problems):
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise ScriptException(proc.returncode, stdout, stderr, script)
    return stdout, stderr

class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        Exception.__init__('Error in script')

def removeDirsInLst(dirLst):
    for dir in dirLst:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        elif os.path.isfile(dir):
            os.remove(dir)
        else:
            print('WARNING: {} not exists, removing passed'.format(dir))

def gifToArr(gifPath):
    with Image.open(gifPath) as img:
        return np.array([np.array(frame.copy().convert('RGB').getdata(),dtype=np.float32).reshape(frame.size[1],frame.size[0],3) for frame in ImageSequence.Iterator(img)])

# def saveGifFromArr(arr, path, fps = 5, vmin = None, vmax = None, axis = 0):
#     """Save 4D Numpy array into gif.
    
#     Last dimension(shape[3]) is regarded as RGB channel.
#     """
    
#     assert(fps >= 1 and axis in [0, 1, 2])
#     arrFormatted = cutArrayToMinMax(arr, min = vmin, max = vmax)
#     if len(arr.shape) == 4:
#         assert(arr.shape[3] == 3)
#         arrFormatted = min_max_scale(arr, 0, 255).astype(np.uint8)
#     elif len(arr.shape) == 3:
#         arrFormatted = visualization.convert_3Darray_to_4DarrayRGB(arr)
#     if vmin is not None:
#         arrFormatted_mask = np.where(arrFormatted >= vmin, 1., 0.)
#     else:
#         arrFormatted_mask = np.where(arrFormatted >= 1e-8, 1., 0.)
#     clip_mask = mpy.VideoClip(make_frame=lambda  t: getSlicesV2(arrFormatted_mask, {axis: int(t)}), duration=arr.shape[0], ismask=True)
#     # clip = mpy.VideoClip(lambda t: arrFormatted[int(t), :, :, :], duration=(arr.shape[0]) / float(fps))
#     clip = mpy.VideoClip(lambda t: getSlicesV2(arrFormatted, {axis: int(t)}), duration = arr.shape[0])
#     clip.set_mask(clip_mask)
#     clip.speedx(fps).write_gif(path, fps = fps)

# def convert_3Darray_to_4DarrayRGB(arr_3D, vmin = None, vmax = None, cmap = plt.get_cmap('jet')):
#     if vmin is not None and vmax is not None:
#         assert(vmax >= vmin)
#     if vmin is None and vmax is None:
#         arr_3D_copied = np.copy(arr_3D)
#     else:
#         arr_3D_copied = cutArrayToMinMax(arr_3D, min = vmin, max = vmax)
#     arr_3D_copied = min_max_scale(arr_3D_copied)
#     arr_4D = cmap(arr_3D_copied, bytes = True)
#     arr_4D = np.delete(arr_4D, 3, 3)
#     return arr_4D

def get_proportional_ranked_value(amount_arr, counter_arr = None, proportion = 0.1):
    """Get ranked values among non-zero values from the numpy array.

    Parameters
    ----------
    proportion : float in [0., 1.] or int
        If float, then proportion of rank from the smallest value. For example, proportion = 0.02 gives top 2% from largest value. If float, then it is direct rank from the largest value.
    
    Examples
    --------
    print(get_proportional_ranked_value(np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), proportion = 0.3))
    >>> 8
    
    """
    
    # assert(proportion >= 0. and proportion <= 1.)
    if counter_arr is not None:
        # num_of_nonzero = np.count_nonzero(np.where(counter_arr >= 1e-8, 1., 0.))
        num_of_nonzero = np.count_nonzero(counter_arr)
    else:
        # num_of_nonzero = np.count_nonzero(np.where(amount_arr >= 1e-8, 1., 0.))
        num_of_nonzero = np.count_nonzero(amount_arr)
    if 0. <= proportion <= 1.:
        rank = int(proportion * num_of_nonzero)
    else:
        rank = proportion
    return np.partition(amount_arr.flatten(), -1 * rank)[-1 * rank]

def collect_idx_of_dense_slices_along_axis(arr, axis, get_dense_slice_threshold = 0, nonzero_threshold = 1e-8, return_first_slice_when_empty = True):
    """Get list of indices of dense(the number of nonzero entries are larger than get_dense_slice_threshold) slices along axis
    
    """

    nonzero_idx_list = []
    if get_dense_slice_threshold > 0:
        for idx in range(arr.shape[axis]):
            sliced = get_slices_with_idc(arr, {axis:[idx]})
            if np.count_nonzero(np.where(sliced > nonzero_threshold, 1., 0.)) >= get_dense_slice_threshold:
                nonzero_idx_list.append(idx)
        if return_first_slice_when_empty and len(nonzero_idx_list) == 0:
            return [0]
        else:
            return nonzero_idx_list
    else:
        return list(range(arr.shape[axis]))

def evaluate_mineral_correlation(geoTensor):
    xcoord, ycoord, zcoord = np.nonzero(geoTensor.counterTensorDict['au'])
    auC = geoTensor.counterTensorDict['au']
    agC = geoTensor.counterTensorDict['ag']
    pbC = geoTensor.counterTensorDict['pb']
    cuC = geoTensor.counterTensorDict['cu']
    znC = geoTensor.counterTensorDict['zn']
    asC = geoTensor.counterTensorDict['as']
    hgC = geoTensor.counterTensorDict['hg']
    sbC = geoTensor.counterTensorDict['sb']
    AU_vals = []
    AG_vals = []
    PB_vals = []
    CU_vals = []
    ZN_vals = []
    AS_vals = []
    HG_vals = []
    SB_vals = []

    for x,y,z in zip(xcoord,ycoord,zcoord):
        if auC[x,y,z] > 0 and agC[x,y,z] > 0 and pbC[x,y,z] > 0 and cuC[x,y,z] > 0 and znC[x,y,z] > 0 and asC[x,y,z] > 0 and hgC[x,y,z] > 0 and sbC[x,y,z] > 0:
            AU_vals.append(geoTensor.dataTensorDict['au'][x][y][z])
            AG_vals.append(geoTensor.dataTensorDict['ag'][x][y][z])
            PB_vals.append(geoTensor.dataTensorDict['pb'][x][y][z])
            CU_vals.append(geoTensor.dataTensorDict['cu'][x][y][z])
            ZN_vals.append(geoTensor.dataTensorDict['zn'][x][y][z])
            AS_vals.append(geoTensor.dataTensorDict['as'][x][y][z])
            HG_vals.append(geoTensor.dataTensorDict['hg'][x][y][z])
            SB_vals.append(geoTensor.dataTensorDict['sb'][x][y][z])

    AU_vals = np.asarray(AU_vals)
    AG_vals = np.asarray(AG_vals)
    PB_vals = np.asarray(PB_vals)
    CU_vals = np.asarray(CU_vals)
    ZN_vals = np.asarray(ZN_vals)
    AS_vals = np.asarray(AS_vals)
    HG_vals = np.asarray(HG_vals)
    SB_vals = np.asarray(SB_vals)
    
    # calculate covariance
    X = np.stack((AU_vals, AG_vals, PB_vals, CU_vals, ZN_vals, AS_vals, HG_vals, SB_vals), axis=0)
    corr = np.corrcoef(X)
    print(corr.shape)
    np.savetxt("mineral_correlation.csv", corr, delimiter=",")
def categoricalTensor(dataTensorDict, counterTensorDict, numberOfClasses, get_avg = False):
    for key in dataTensorDict.keys():
        assert(len(dataTensorDict[key]) == len(counterTensorDict[key]))
        shape_ = dataTensorDict[key].shape
    shape_ = list(shape_)
    shape_.append(numberOfClasses)
    shape_ = tuple(shape_) # convert 3D shape to 4D shoape with number of classes
    mineralLst = dataTensorDict.keys()
    categoricalTensorDict = {}
    range_list_dict = {}
    avg_list_dict = {}
    for mineral in mineralLst:
        range_list_dict[mineral] = [0] * numberOfClasses
        avg_list_dict[mineral] = [0] * numberOfClasses
    for mineral in mineralLst:
        categoricalTensorDict[mineral] = np.zeros(shape_)
    amountsDict = {}
    for mineral in mineralLst:
        amountsDict[mineral] = np.sort(dataTensorDict[mineral][counterTensorDict[mineral].nonzero()])
        for classIdx in range(numberOfClasses):
            range_list_dict[mineral][classIdx] = amountsDict[mineral][int(amountsDict[mineral].shape[0] / numberOfClasses) * classIdx]
            avg_list_dict[mineral][classIdx] = amountsDict[mineral][int(amountsDict[mineral].shape[0] / numberOfClasses) * classIdx + int(amountsDict[mineral].shape[0] * 0.5 / numberOfClasses)]
    # 0.0 ~ 0.7 -> class 0, X , 0.7 ~ 3.2 -> class 0
    #
    for mineral in dataTensorDict.keys():
        nonzerosIndices = counterTensorDict[mineral].nonzero()
        for positionIdx in range(nonzerosIndices[0].shape[0]):
            for category in range(numberOfClasses):
                if category == len(range_list_dict[mineral]) - 1 or dataTensorDict[mineral][nonzerosIndices[0][positionIdx], nonzerosIndices[1][positionIdx], nonzerosIndices[2][positionIdx]] < range_list_dict[mineral][category + 1]:
                    categoricalTensorDict[mineral][nonzerosIndices[0][positionIdx], nonzerosIndices[1][positionIdx], nonzerosIndices[2][positionIdx], category] = 1
                    break
    if get_avg:
        return categoricalTensorDict, avg_list_dict
    else:                        
        return categoricalTensorDict

def get_dummy_arr_list(shape_, functions_list, relation_matrix, origin = 'center', noise = {'mean':0., 'std': 0.1}, min_max_scaled = True):
    """
    
    Parameters
    ----------
    origin : str or length-3 list
        Substract origin from input index of function.
    relation_matrix : 2-D array
        i-th row and j-th column indicates the weights or influence of j-th array on i-th array.
    
    """
    if origin == 'center':
        translate_substraction_ijk = [shape_[0] // 2, shape_[1] // 2, shape_[2] // 2,]
    else:
        translate_substraction_ijk = deepcopy(origin)

    assert(len(functions_list) == relation_matrix.shape[0])
    assert(len(relation_matrix.shape) == 2 and relation_matrix.shape[0] == relation_matrix.shape[1])
    result_arr_list_before_relation = []
    for function, idx in zip(functions_list, range(len(functions_list))):
        arr_of_function = np.empty(shape_)
        for i in range(shape_[0]):
            for j in range(shape_[1]):
                for k in range(shape_[2]):
                    arr_of_function[i, j, k] = function(i - translate_substraction_ijk[0], j - translate_substraction_ijk[1], k - translate_substraction_ijk[2])
        arr_of_function = arr_of_function + np.random.normal(loc = noise['mean'], scale = noise['std'], size = shape_)
        result_arr_list_before_relation.append(arr_of_function)
    result_arr_list_after_relation = []
    for arr_before_relation, row_idx in zip(result_arr_list_before_relation, range(len(result_arr_list_before_relation))):
        array_after_relation = np.copy(arr_before_relation)
        for col_idx in range(relation_matrix.shape[1]):
            if col_idx != row_idx:
                array_after_relation += relation_matrix[row_idx, col_idx] * result_arr_list_before_relation[col_idx]
        result_arr_list_after_relation.append(min_max_scale(array_after_relation))
    return result_arr_list_after_relation

def get_y_x_dictionary(x, y, num_samples_each_category = 1000):
    assert(x.shape[0] == y.shape[0])
    y_x_dict = {}
    for idx in range(x.shape[0]):
        if y[idx] in y_x_dict.keys():
            if len(y_x_dict[y[idx]]) < num_samples_each_category:
                y_x_dict[y[idx]].append(x[idx])
        else:
            y_x_dict[y[idx]] = []
    for y_ in y_x_dict.keys():
        if len(y_x_dict[y_]) < num_samples_each_category:
            raise Exception(f"The number of samples for category {y_x_dict[y_]} = {len(y_x_dict[y_])} is not enough for required {num_samples_each_category}")
        else:
            y_x_dict[y_] = np.stack(y_x_dict[y_], axis = 0).astype(np.float32)
    return y_x_dict

def random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item= True):
    """

    Examples
    --------
    item_length_dict = {0: 3, 1: 4, 4: 2}\n
    pick_keep_probs_dict = {-1: 0.3, 0: 0.5}\n
    keep_non_prob_item = True\n
    print(random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item))\n
    {0: [2, 1], 1: [1, 2], 4: [1]}\n
    """

    picked_idc_dict = {}
    for idx_0 in item_length_dict.keys():
        picked_idc_dict[idx_0] = []
        done_idx= []
        for key, prob in pick_keep_probs_dict.items():
            key_positive = key if key>= 0 else item_length_dict[idx_0] + key
            if key_positive in done_idx:
                raise Exception(f"Ambiguous pick with index: {key}")
            done_idx.append(key_positive)
            rand_num= random()
            if rand_num<= prob:
                picked_idc_dict[idx_0].append(key_positive)
        if keep_non_prob_item:
            for idx_1 in range(item_length_dict[idx_0]):
                if idx_1 not in done_idx and idx_1 not in picked_idc_dict[idx_0]:
                    picked_idc_dict[idx_0].append(idx_1)
    return picked_idc_dict

def delete_items_from_list_with_indices(list_to_filter, indices, keep_not_remove = False):
    """Delete items from list with indices
    
    Examples
    --------
    delete_items_from_list_with_indices(['a', 'b', 7], [0, 2])
        : ['b']
    """
    
    base_idx = 0
    sorted_indices = sorted(indices, reverse = False)
    sorted_indices_to_remove = []
    if keep_not_remove:
        for i in range(len(list_to_filter)):
            if i not in sorted_indices:
                sorted_indices_to_remove.append(i)
    else:
        sorted_indices_to_remove = deepcopy(sorted_indices)
    list_to_filter_copied = deepcopy(list_to_filter)
    for idx in sorted_indices_to_remove:
        del list_to_filter_copied[idx - base_idx]
        base_idx += 1
    return list_to_filter_copied

def cutArrayToMinMax(arr, min = None, max = None):
    """Cut given array with given ceil and floor
    
    This is not scaling but cutting.
    """

    if min is not None and max is not None:
        assert(max >= min)
        return np.where(arr > max, max, np.where(arr < min, min, arr))
    elif min is None and max is not None:
        return np.where(arr > max, max, arr)
    elif min is not None and max is None:
        return np.where(arr < min, min, arr)
    else:
        return arr

def access_with_list_of_keys_or_indices(container_tobe_accessed, list_of_keys_or_indices):
    """Deprecated : Use containers.access_with_list_of_keys_or_indices.
    
    Helper recursive function for access_with_list_of_keys_or_indices function. 201115 : Name changed from access_with_list_of_keys_or_indices_rec to access_with_list_of_keys_or_indices.

    Examples
    --------
    print(access_with_list_of_keys_or_indices({"a": 3, "b": [4, {5: [6, 7]}]}, ["b", 1, 5, 0]))
        >>> 6
    """

    key_or_index = list_of_keys_or_indices[0]
    remaining_list_of_keys_or_indices = list_of_keys_or_indices[1:]
    if isinstance(container_tobe_accessed[key_or_index], list) or isinstance(container_tobe_accessed[key_or_index], dict) or isinstance(container_tobe_accessed[key_or_index], tuple): ## next child node is list/dict/tuple.
        if len(remaining_list_of_keys_or_indices) > 0: ## We need to go one step more.
            return access_with_list_of_keys_or_indices(container_tobe_accessed[key_or_index], remaining_list_of_keys_or_indices)
        else: ## No more to go.
            return container_tobe_accessed[key_or_index]
    else: ## Next child node is not a container: list/dict/tuple, so ends here.
        return container_tobe_accessed[key_or_index]

# def access_with_list_of_keys_or_indices(container_tobe_accessed, list_of_keys_or_indices):
#     """This is not in-place function. This function returns pointer to an element, not an element copied. 
    
#     Examples
#     --------
#     test_dict = {'hi':[1, {'hello': [3, 4]}], 'end': [3, 6]}\n
#     print(access_with_list_of_keys_or_indices(test_dict, ['hi', 1, 'hello', 1]))
#         >>> 4
#     """

#     list_of_keys_or_indices_copied = deepcopy(list_of_keys_or_indices)
#     return access_with_list_of_keys_or_indices_rec(container_tobe_accessed, list_of_keys_or_indices_copied)

def get_paths_to_leaves_rec(container, paths):
    """Deprecated : Use containers.get_paths_to_leaves_rec
    
    Helper recursive function for get_paths_to_leaves"""

    result_paths = []
    for path in paths:
        current_path = deepcopy(path)
        current_container = access_with_list_of_keys_or_indices(container, path)
        current_indices = []
        if any([isinstance(current_container, list), isinstance(current_container, tuple)]):
            for i in range(len(current_container)):
                current_indices.append([i])
        elif isinstance(current_container, dict):
            for key in current_container.keys():
                current_indices.append([key])
        else: ## base case
            result_paths.append(current_path)
        if any([isinstance(current_container, list), isinstance(current_container, tuple), isinstance(current_container, dict)]):
            sub_paths = get_paths_to_leaves_rec(current_container, current_indices)
            for sub_path in sub_paths:
                result_paths.append(current_path + sub_path)
    return result_paths             

def get_paths_to_leaves(container):
    """Deprecated : Use containers.get_paths_to_leaves
    
    Get paths to leaves from nested dictionary or list.
    
    Parameters
    ----------
    container : dict or list or tuple

    Examples
    --------
    test_dict = {'hi':[1, {'hello': [3, 4]}], 'end': [3, 6]}\n
    print(get_paths_to_leaves(test_dict))
        [['hi', 0], ['hi', 1, 'hello', 0], ['hi', 1, 'hello', 1], ['end', 0], ['end', 1]]
    """

    assert(any([isinstance(container, list), isinstance(container, tuple), isinstance(container, dict)]))
    current_indices = []
    if any([isinstance(container, list), isinstance(container, tuple)]):
        for i in range(len(container)):
            current_indices.append([i])
    elif isinstance(container, dict):
        for key in container.keys():
            current_indices.append([key])
    return get_paths_to_leaves_rec(container, current_indices)

def index_arr_with_arraylike(arr_tobe_indexed, index_arraylike):
    """Index Numpy array with indices
    
    Examples
    --------
    test_arr = np.ones((3, 3))
    print('1: ', index_arr_with_arraylike(test_arr, np.array([1, 2])))
    print('2: ', index_arr_with_arraylike(test_arr, [0]))
    print('3: ', test_arr)
        1:  1.0
        2:  [1. 1. 1.]
        3:  [[1. 1. 1.]
        [1. 1. 1.]
        [1. 1. 1.]]
    """

    result = arr_tobe_indexed ## shallow copy
    if isinstance(index_arraylike, np.ndarray):
        assert(len(index_arraylike.shape) == 1)
        for i in range(index_arraylike.shape[0]):
            result = result[index_arraylike[i]]
    elif isinstance(index_arraylike, (list, tuple)):
        for i in index_arraylike:
            result = result[i]
    else:
        raise Exception("Unsupported type of index_arraylike")
    return result

def get_smallest_largest_idc(input_list, num_idc = 1, smallest = True):
    """Get the num_idc smallest items
    
    Examples
    --------
    test_list = [3, 4.5, 2, 9, 4]\n
    print(get_smallest_largest_idc(test_list, 2))
        : [2, 0]
    """

    assert(num_idc > 0)
    temp_list = []
    for elements, index in zip(input_list, range(len(input_list))):
        temp_list.append([elements, index])
    temp_list.sort(key = lambda x: x[0], reverse = not smallest)
    result_idc_list = []
    for i in range(num_idc):
        result_idc_list.append(temp_list[i][1])
    return result_idc_list

def get_top_n_indices(array_like, n = 1, from_largest = True):
    """Get the indices of top-n elements largest(default) or smallest
    
    Examples
    --------
    print(get_top_n_indices(array_like = [3, 5, 1, 7, 3, 5], n = 3, from_largest = True))
        : [3 1 5]
    print(get_top_n_indices(array_like = np.array([3, 5, 1, 7, 3, 5]), n = 3, from_largest = False))
        : [2 4 0]
    """
    
    if not isinstance(array_like, np.ndarray):
        array_input = np.array(array_like)
        assert(len(array_input.shape) == 1)
    else:
        assert(len(array_like.shape) == 1)
        array_input = np.copy(array_like)
    factor_from_largest = -1 if from_largest else +1
    if from_largest:
        ind = np.argpartition(array_input, n * factor_from_largest)[n * factor_from_largest:]
    else:
        ind = np.argpartition(array_input, n * factor_from_largest)[:n * factor_from_largest]
    return ind[np.argsort(factor_from_largest * array_input[ind])]

def add_column_conditional(df, column_bemapped, mapping, new_column_name = 'added'):
    """
    
    Parameters
    ----------
    mapping: dict, list, function
        Condition about how to map the element to new value of new column. If list, unique elements of column_bemapped to the elements of list, and if the number of unique elements exceed the length of list, the elements of list is rotated.

    Returns
    -------
    df_copied : pandas df
        dataframe with added column
    """

    assert(new_column_name not in df.columns)
    df_copied = df.copy()
    if callable(mapping):
        raise(Exception(NotImplementedError))
        return 0
    if isinstance(mapping, list):
        unique_names_list = list(df_copied[column_bemapped].unique())
        name_map_dict = {}
        for name, idx in zip(unique_names_list, range(len(unique_names_list))):
            name_map_dict[name] = mapping[idx % len(mapping)]
    elif isinstance(mapping, dict):
        name_map_dict = deepcopy(mapping)
    ## Do mapping == list or dict case
    df_copied[new_column_name] = df_copied[column_bemapped].map(name_map_dict)
    return df_copied

def get_current_utc_timestamp():
    """Get current utc time in your local time.
    
    Examples
    --------
    get_current_utc_timestamp()
        >>> 1605306120.187412
    """

    dt = datetime.now()
    utc_time = dt.replace(tzinfo = timezone.utc)
    return utc_time.timestamp()

if __name__ == '__main__':
    pass
    print(get_proportional_ranked_value(np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), proportion = 0.3))