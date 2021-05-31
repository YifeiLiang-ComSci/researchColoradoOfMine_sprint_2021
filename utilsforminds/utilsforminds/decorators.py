import utilsforminds
import functools
import numpy as np
import utilsforminds.helpers as helpers
from inspect import signature

def signature_multi_binding(new_old_signature_dict):
    def decorator(func):
        @functools.wraps(func)
        def decorator_inner(*args, **kwargs):
            processed_news = []
            for new, old in new_old_signature_dict.items():
                raise Exception(NotImplementedError)

def redirect_function(module, func_name = None):
    """Redirect the function to another function in another module"""

    def decorator_redirect_function(func):
        @functools.wraps(func)
        def wrapper_redirect_function(*args, **kwargs):
            if func_name is None:
                return getattr(module, func.__name__)(*args, **kwargs)
            else:
                return getattr(module, func_name)(*args, **kwargs)
        return wrapper_redirect_function
    return decorator_redirect_function

def check_bad_values_in_numpy_arr(check_list = ['nan', 'inf']):
    """Check whether input Numpy arguments contain invalid values
    
    Examples
    --------
    test_arr_1 = np.ones((3, 3))
    test_arr_2 = np.ones((3, 3)) + 2.
    test_arr_2[1, 2] = np.inf
    print(add_arr(test_arr_1, name = 'hello', arr_2= test_arr_2))
        :Exception has occurred: Exception
        Invalid values found
    """

    def decorator_check_bad_values_in_numpy_arr(func):
        @functools.wraps(func)
        def wrapper_check_bad_values_in_numpy_arr(*args, **kwargs):
            for arg in args:
                if isinstance(arg, np.ndarray):
                    bool_arr = np.full(arg.shape, False, dtype= bool)
                    if 'nan' in check_list:
                        bool_arr = bool_arr + np.isnan(arg)
                    if 'inf' in check_list:
                        bool_arr = bool_arr + np.isinf(arg)
                    if np.any(bool_arr):
                        indices_arr = np.argwhere(bool_arr)
                        for i in range(min(10, indices_arr.shape[0])):
                            print(f"WARNING: Invalid value: {helpers.index_arr_with_arraylike(arg, indices_arr[i])} is found at indices: {indices_arr[i]}")
                        raise Exception("Invalid values found")
            for arg in kwargs.values():
                if isinstance(arg, np.ndarray):
                    bool_arr = np.full(arg.shape, False, dtype= bool)
                    if 'nan' in check_list:
                        bool_arr = bool_arr + np.isnan(arg)
                    if 'inf' in check_list:
                        bool_arr = bool_arr + np.isinf(arg)
                    if np.any(bool_arr):
                        indices_arr = np.argwhere(bool_arr)
                        for i in range(min(10, indices_arr.shape[0])):
                            print(f"WARNING: Invalid value: {helpers.index_arr_with_arraylike(arg, indices_arr[i])} is found at indices: {indices_arr[i]}")
                        raise Exception("Invalid values found")
            return func(*args, **kwargs)
        return wrapper_check_bad_values_in_numpy_arr
    return decorator_check_bad_values_in_numpy_arr

if __name__ == "__main__":
    pass
    # @check_bad_values_in_numpy_arr()
    # def add_arr(arr_1, arr_2, name):
    #     print(name)
    #     return arr_1 + arr_2

    # test_arr_1 = np.ones((3, 3))
    # test_arr_2 = np.ones((3, 3)) + 2.
    # # test_arr_2[1, 2] = np.inf
    # print(add_arr(test_arr_1, name = 'hello', arr_2= test_arr_2))