import numpy as np
import utilsforminds.helpers as helpers
from copy import deepcopy
import numbers
import math
from keras import backend as K
import tensorflow as tf

def p_exponent_matrix(arr, p):
    """
    
    Examples
    --------
    self.diagonals['D_7'] = (self.p / 2) * helpers.p_exponent_matrix(W_p1p @ W_p1p.transpose() + self.delta * np.eye(self.d), (self.p / 2 - 1))
    """
    assert(len(arr.shape) == 2)

    #%% Using SVD
    u, s, v = np.linalg.svd(arr, full_matrices=False)
    return u @ np.diag(s ** p) @ v

    #$$ Using Igenvalue decomposition
    # w, v = np.linalg.eig(arr)
    # return v @ np.diag(w ** p) @ v.transpose()

def get_norm_from_matrix(arr, under_p_1, under_p_2):
    """
    
    Examples
    --------
    testArr_1 = np.array(list(range(6))).reshape(2,3)\n
    print(helpers.get_norm_from_matrix(testArr_1, 2, 2))
        => 7.416198487095663
    """

    summed = 0
    for i in range(arr.shape[0]):
        summed += np.sum(arr[i, :] ** under_p_1) ** (under_p_2 / under_p_1)
    return summed ** (1 / under_p_2)

def get_RMSE(y_true, y_pred):
    assert(y_true.shape == y_pred.shape)
    return (np.sum((y_true - y_pred) ** 2) / y_true.shape[0]) ** (1/2.)

def get_RMSE_keras(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def statistics_across_containers(containers_list, kind_of_stat = 'std'):
    """Get statistic values across the list of the leaves of the nested containers whose shapes are same
    
    Parameters
    ----------
    containers_list : list
        List of containers, say [container_1, container_2, ..., container_n]. Here container_1, container_2, ..., container_n can be nested dict or list or tuple, but their nested structure should be same.
    
    Returns
    -------
    : container
        Container of statistic values

    Examples
    --------
    test_containers_list = [{'a': [1, 2.5, 3], 'b': {'c': [2, 4], 'd': 5}}, {'a': [2, 3.5, 4], 'b': {'c': [3, 5], 'd': 6}}]\n
    print(statistics_across_containers(test_containers_list, kind_of_stat = 'mean'))
        {'a': [1.5, 3.0, 3.5], 'b': {'c': [2.5, 4.5], 'd': 5.5}}
    """

    assert(isinstance(containers_list, list))
    result_container = deepcopy(containers_list[0])
    paths_to_leaves = deepcopy(helpers.get_paths_to_leaves(result_container))
    for path in paths_to_leaves:
        container_last_one = helpers.access_with_list_of_keys_or_indices(result_container, path[:-1])
        container_last_one[path[-1]] = []
    
    for container in containers_list:
        for path in paths_to_leaves:
            helpers.access_with_list_of_keys_or_indices(result_container, path).append(helpers.access_with_list_of_keys_or_indices(container, path))
    
    for path in paths_to_leaves:
        container_last_one = helpers.access_with_list_of_keys_or_indices(result_container, path[:-1])
        elements_across_containers_list = helpers.access_with_list_of_keys_or_indices(result_container, path)
        is_all_elements_are_number = True
        for element in elements_across_containers_list:
            assert(not isinstance(element, list))
            if not isinstance(element, numbers.Number):
                is_all_elements_are_number = False
                break
        if is_all_elements_are_number:
            if kind_of_stat == 'std':
                container_last_one[path[-1]] = np.std(np.array(elements_across_containers_list))
            elif kind_of_stat == 'mean':
                container_last_one[path[-1]] = np.mean(np.array(elements_across_containers_list))
            else:
                raise Exception(f"Unsupported kind_of_stat: {kind_of_stat}")
    
    return result_container

def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    original ref: http://randlet.com/blog/python-significant-figures-format/
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

def is_converged(loss_lst, consecutive_trends = 4, comparison_ratio = 1.0, check_start = 1, debug = False):
    """Check whether the list of loss given is converged or not.
    
    The smaller comparison_ratio and the larger consecutive_trends result the stricter checking.

    """

    #%% For debugging
    if debug and len(loss_lst) > 1:
        return True

    if check_start is not None:
        if len(loss_lst) < consecutive_trends + check_start + 1:
            return False
        else:
            gradient_avg = 0
            for i in range(check_start, len(loss_lst) - consecutive_trends):
                gradient_avg += abs(loss_lst[i + 1] - loss_lst[i])
            gradient_avg = gradient_avg / (len(loss_lst) - consecutive_trends - check_start)
            for past in range(1, consecutive_trends):
                if (abs(loss_lst[- past] - loss_lst[- past - 1]) > gradient_avg * comparison_ratio):
                    return False
            return True
    # else:
    #     if len(loss_lst) < consecutive_trends + 1:
    #         return False
    #     else:
    #         for past in range(1, consecutive_trends):
    #             if loss_lst[- past] < loss_lst[- past - 1] - (loss_lst[- past - 1] * (1 - comparison_ratio))

def sparse_group_lasso_function_object(reg_factor = 1e5):
    """Noor's code for defining the regularizer"""

    def sparse_group_lasso(weights):
        include_group_norm = True
        if not include_group_norm:
            return tf.constant(0.0)
        vector_of_l2_norms = tf.norm(weights, ord=2, axis=1)
        group_norm = tf.norm(vector_of_l2_norms, ord=1)
        lasso = tf.norm(weights, ord=1)
        return reg_factor * (group_norm + lasso)
    return sparse_group_lasso

def mean(numbers, default = None):
    if default is None:
        assert(len(numbers) > 0)
        return sum(numbers) / len(numbers)
    else:
        if len(numbers) > 0: return sum(numbers) / len(numbers)
        else: return default

def std(numbers, default = None):
    if default is None or len(numbers) > 0:
        mean_ = mean(numbers)
        squared_diff_sum = 0.
        for number in numbers:
            squared_diff_sum += (number - mean_) ** 2.
        return (squared_diff_sum / len(numbers)) ** 0.5
    else:
        return default

def is_number(target):
    if isinstance(target, (int, float, complex)) and not isinstance(target, bool):
        return True
    else:
        return False
def get_new_weight_based_loss_trends(losses, current_weight, mean_before_losses_step_backwards = 6, mean_after_losses_step_backwards = 3, factor_weight_change_to_loss_change = +0.1, kind = 'arithmetic', max_weight = None, min_weight = None, do_nothing = False, verbose = False):
    """
    
    Parameters
    ----------
    factor_weight_change_to_loss_change : float
        If factor_weight_change_to_loss_change is positive, then weight IS INCREASED as loss IS INCREASED.

    Examples
    --------
    print(get_new_weight_based_loss_trends([1, 2, 1, 2, 1, 2, 3, 4, 5], 0.1))
        == 0.20714285714285716
    """
    if do_nothing or len(losses) < mean_before_losses_step_backwards + mean_after_losses_step_backwards:
        return current_weight
    
    loss_mean = mean(losses)
    loss_before_local_mean = mean(losses[-(mean_before_losses_step_backwards + mean_after_losses_step_backwards):-mean_after_losses_step_backwards])
    loss_after_local_mean = mean(losses[-mean_after_losses_step_backwards:])
    loss_change = (loss_after_local_mean - loss_before_local_mean) / abs(loss_mean)

    if kind == "arithmetic":
        new_weight = current_weight + loss_change * factor_weight_change_to_loss_change
        if max_weight is not None:
            new_weight = min(new_weight, max_weight)
        if min_weight is not None:
            new_weight = max(min_weight, new_weight)
    else:
        raise Exception(f"Unsupported kind: {kind}")
    if verbose:
        print(f"Weight changes from {current_weight} to {new_weight}")
    
    ## keep the sign
    if current_weight * new_weight < 0:
        return current_weight
    else:
        return new_weight



if __name__ == '__main__':
    pass
    # test_containers_list = [{'a': [1, 2.5, 3], 'b': {'c': [2, 4], 'd': 5}}, {'a': [2, 3.5, 4], 'b': {'c': [3, 5], 'd': 6}}]
    # print(statistics_across_containers(test_containers_list, kind_of_stat = 'mean'))
    # print(to_precision(1.23,4))
    # print(get_new_weight_based_loss_trends([1, 2, 1, 2, 1, 2, 3, 4, 5], 0.1))
    print(mean([2.0, 3.0]))
    print(std([2.0, 3.0]))