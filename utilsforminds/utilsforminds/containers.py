from copy import deepcopy
import numpy as np

def get_items_from_list_conditionally(list_, condition_function, whether_deepcopy = False):
    """ This is reference-copy function not value-copy.
    
    Examples
    --------
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4}]\n
    print(get_items_from_list_conditionally(test_list, lambda x: True if x['a'] >= 2 else False))
        [{'a': 3, 'b': 2}, {'a': 2, 'b': 4}]
    """

    assert(type(list_) == type([]))
    result_collection = []
    for item in list_:
        if condition_function(item):
            result_collection.append(deepcopy(item)) if whether_deepcopy else result_collection.append(item)
    return result_collection

def merge_two_dicts_leaves_rec(dict_1, dict_2):
    """ Merge two dictionaries recursively.

    Examples
    --------
    test_dict_1 = {"n": ["a", "b", "c"], "m": {"hi": 4}, 1: ["h", "i"], "dup": {"ace": {1: 2, 3: 4}}, "none": None}
    test_dict_2 = {"n": ["a", "d"], "m": {"hi" : 2, "hello": {3: 2}}, 2: 4, "dup": {"ace": {1: {4: 5, 6: 7}, 3: None}}, "none": 4}
    print(merge_two_dicts_leaves_rec(test_dict_1, test_dict_2))
        >>> {'n': ['a', 'd'], 'm': {'hi': 2, 'hello': {3: 2}}, 2: 4, 'dup': {'ace': {1: {4: 5, 6: 7}, 3: None}}, 'none': 4, 1: ['h', 'i']}
    """

    if dict_1 is None:
        return dict_2
    if dict_2 is None:
        return dict_1
    assert(isinstance(dict_1, dict) and isinstance(dict_2, dict))
    merged_dict = {}
    for dict_2_key in dict_2.keys():
        if isinstance(dict_2[dict_2_key], dict) and dict_2_key in dict_1.keys() and isinstance(dict_1[dict_2_key], dict):
            merged_dict[dict_2_key] = merge_two_dicts_leaves_rec(dict_1[dict_2_key], dict_2[dict_2_key])
        else:
            merged_dict[dict_2_key] = dict_2[dict_2_key]
    for dict_1_key in dict_1.keys():
        if dict_1_key not in merged_dict.keys():
            merged_dict[dict_1_key] = dict_1[dict_1_key]
    return merged_dict
            

def merge_dictionaries(list_of_dicts : list, use_last_when_overlapped = True, recursive_overwritting = True):
    """ Merge dictionaries from the root to leaves, recursively.

    Always deepcopy because of nature of dict.update method. -> Changed to shallowcopy 200910.

    Examples
    --------
    test_list = [{'a':1, 'b':2, 'c':9}, {'a':3, 'b':2}, {'a':2, 'b':4}]\n
    merge_dictionaries(test_list, use_last_when_overlapped = True)\n
        >>> {'a': 2, 'b': 4, 'c': 9}
    merge_dictionaries(test_list, use_last_when_overlapped = False)\n
        >>> {'a': 1, 'b': 2, 'c': 9}

    Example of shallow copy:
    test_dict_1 = {"a": 1, "b": 2, "c": 3}\n
    test_dict_2 = {"b": 7}\n
    print(merge_dictionaries([test_dict_1, test_dict_2]))
        >>> {'b': 7, 'a': 1, 'c': 3}
    print(test_dict_1)
        >>> {'a': 1, 'b': 2, 'c': 3}
    print(test_dict_2)
        >>> {'b': 7}
    """
    
    merged_dict = {}
    if use_last_when_overlapped:
        range_to_iter = range(len(list_of_dicts))
    else:
        range_to_iter = range(len(list_of_dicts) -1, -1, -1)
    for dict_idx in range_to_iter:
        # if list_of_dicts[dict_idx] is not None:
        #     assert(type(list_of_dicts[dict_idx]) == type({}))
        #     merged_dict.update(list_of_dicts[dict_idx])
        merged_dict = merge_two_dicts_leaves_rec(merged_dict, list_of_dicts[dict_idx])
    return merged_dict

def merge_lists(list_of_lists : list, use_last_when_overlapped = True):
    """Merge lists through deepcopy

    Examples
    --------
    test_list = [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]
    print(merge_lists(test_list, use_last_when_overlapped = True))
        >>> ['h', 'i', 'f', 'g']
    print(merge_lists(test_list, use_last_when_overlapped = False))
        >>> ['a', 'b', 'c', 'g']
    """

    merged_list = []
    if use_last_when_overlapped:
        range_to_iter = range(len(list_of_lists) -1, -1, -1)
    else:
        range_to_iter = range(len(list_of_lists))
    for list_idx in range_to_iter:
        if list_of_lists[list_idx] is not None:
            assert(type(list_of_lists[list_idx]) == type([]))
            for list_idx_idx in range(len(merged_list), len(list_of_lists[list_idx])): ## append only when there are more elements in list_of_lists[list_idx].
                merged_list.append(list_of_lists[list_idx][list_idx_idx])
    return merged_list

def copy_dict_and_delete_element(dict_to_copy, list_of_keys_to_delete):
    """
    
    Examples
    --------
    test_dict = {"n": ["a", "b", "c"], "m": ["d", "e", "f", "g"], 1: ["h", "i"]}
    print(copy_dict_and_delete_element(dict_to_copy = test_dict, list_of_keys_to_delete = ["n", 1]))
        >>> {'m': ['d', 'e', 'f', 'g']}
    """

    result = deepcopy(dict_to_copy)
    for key in list_of_keys_to_delete:
        if key in result.keys():
            del result[key]
    return result

def squeeze_list_of_numbers_with_average_of_each_range(list_of_numbers, num_points_in_list_out = 100):
    """
    
    Examples
    --------
    test_list = list(range(5))
    print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 2))
        >>> [0.5, 2.5]
    test_list = list(range(20))
    print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 5))
        >>> [1.5, 5.5, 9.5, 13.5, 17.5]
    test_list = list(range(19))
    print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 5))
        >>> [1.0, 4.0, 7.0, 10.0, 13.0]
    """

    if len(list_of_numbers) <= num_points_in_list_out:
        return deepcopy(list_of_numbers)
    else:
        list_out = []
        num_numbers_in_each_group = len(list_of_numbers) // num_points_in_list_out
        for group_idx in range(num_points_in_list_out):
            average = sum(list_of_numbers[group_idx * num_numbers_in_each_group: (group_idx + 1) * num_numbers_in_each_group]) / num_numbers_in_each_group
            list_out.append(average)
        return list_out

def access_with_list_of_keys_or_indices(container_tobe_accessed, list_of_keys_or_indices):
    """Helper recursive function for access_with_list_of_keys_or_indices function.

    Other than access (read), write is also possible e.g. access_with_list_of_keys_or_indices(test_dict, test_list_2[:-1])[test_list_2[-1]] = 7.
    201115 : Name changed from access_with_list_of_keys_or_indices_rec to access_with_list_of_keys_or_indices.

    Examples
    --------
    test_dict = {"a": 3, "b": [4, {5: [6, 7]}]}\n
    print(access_with_list_of_keys_or_indices(test_dict, ["b", 1, 5, 0]))
        >>> 6
    print(access_with_list_of_keys_or_indices(test_dict, ["b", 1, 5]))
        >>> [6, 7]
    test_list_1 = ["b", 1, 5, 0]\n
    test_list_2 = ["a"]\n
    access_with_list_of_keys_or_indices(test_dict, test_list_1[:-1])[test_list_1[-1]] = 4\n
    access_with_list_of_keys_or_indices(test_dict, test_list_2[:-1])[test_list_2[-1]] = 7\n
    print(test_dict)
        >>> {'a': 7, 'b': [4, {5: [4, 7]}]}
    """

    if len(list_of_keys_or_indices) == 0:
        return container_tobe_accessed
    
    key_or_index = list_of_keys_or_indices[0]
    remaining_list_of_keys_or_indices = list_of_keys_or_indices[1:]
    if isinstance(container_tobe_accessed[key_or_index], list) or isinstance(container_tobe_accessed[key_or_index], dict) or isinstance(container_tobe_accessed[key_or_index], tuple): ## next child node is list/dict/tuple.
        if len(remaining_list_of_keys_or_indices) > 0: ## We need to go one step more.
            return access_with_list_of_keys_or_indices(container_tobe_accessed[key_or_index], remaining_list_of_keys_or_indices)
        else: ## No more to go.
            return container_tobe_accessed[key_or_index]
    else: ## Next child node is not a container: list/dict/tuple, so ends here.
        return container_tobe_accessed[key_or_index]

def get_paths_to_leaves_rec(container, paths, leaf_include_condition = lambda x: True):
    """Helper recursive function for get_paths_to_leaves
    
    Parameters
    ----------
    leaf_include_condition : Callable
        When we arrive the leaf node (leaf is defined as 'not container'), we include current path to the leaf if this condition is satisfied, otherwise we discard current path from the result_paths.
    """

    result_paths = []
    for path in paths:
        ## Get current path and it's container.
        current_path = deepcopy(path) ## e.g. path = [1, 'b', 3].
        current_container = access_with_list_of_keys_or_indices(container, path)

        ## Stack the sub_paths to search more.
        indices_to_search_more = [] ## We continue to search paths from indices_to_search_more.
        
        if is_container(current_container): ## --- recursive case.
            ## Get range to iterate.
            if any([isinstance(current_container, list), isinstance(current_container, tuple)]): list_to_iterate = range(len(current_container))
            elif isinstance(current_container, dict): list_to_iterate = list(current_container.keys())

            ## Iterate and stack first index or key.
            for idx in list_to_iterate:
                if is_container(current_container[idx]): indices_to_search_more.append([idx])
                elif leaf_include_condition(current_container[idx]): result_paths.append(current_path + [idx]) ## When current's child is the base case, this should be found here, and should not be searched anymore.
            
            ## Recursive calls for the childs sub_paths.
            sub_paths = get_paths_to_leaves_rec(current_container, indices_to_search_more, leaf_include_condition = leaf_include_condition) ## Sub_paths from the current nodes (indices_to_search_more).
            for sub_path in sub_paths:
                result_paths.append(current_path + sub_path)
        
        elif leaf_include_condition(current_container): ## --- base case.
            result_paths.append(current_path) ## Here is the leaf, stop searching.
            
    return result_paths             

def get_paths_to_leaves(container, leaf_include_condition = lambda x: True):
    """Get paths to leaves from nested dictionary or list
    
    Parameters
    ----------
    container : dict or list or tuple
        Container to iterate.
    leaf_include_condition : Callable
        When we arrive the leaf node (leaf is defined as 'not container'), we include current path to the leaf if this condition is satisfied, otherwise we discard current path from the result_paths.

    Examples
    --------
    test_dict = {'hi':[1, {'hello': [3, 4]}], 'end': [3, 6], 7: "hey"}\n
    print(get_paths_to_leaves(test_dict))
        >>> [[7], ['hi', 0], ['hi', 1, 'hello', 0], ['hi', 1, 'hello', 1], ['end', 0], ['end', 1]]
    print(get_paths_to_leaves(test_dict, leaf_include_condition= lambda x: True if isinstance(x, int) else False))
        >>> [['hi', 0], ['hi', 1, 'hello', 0], ['hi', 1, 'hello', 1], ['end', 0], ['end', 1]]
    test_list = ["a", {"b": {3: 4}}, [4, 5, [6, 7]], 4]\n
    print(get_paths_to_leaves(test_list))
        >>> [[0], [3], [1, 'b', 3], [2, 0], [2, 1], [2, 2, 0], [2, 2, 1]]
    print(get_paths_to_leaves(test_list, leaf_include_condition= lambda x: True if isinstance(x, int) else False))
        >>> [[1, 'b', 3], [2, 0], [2, 1], [2, 2, 0], [2, 2, 1], [3]]
    """

    assert(any([isinstance(container, list), isinstance(container, tuple), isinstance(container, dict)]))
    final_indices = [] ## For case when the immediate child node is not a container.
    current_indices = []

    ## Get range to iterate.
    if any([isinstance(container, list), isinstance(container, tuple)]): list_to_iterate = range(len(container))
    elif isinstance(container, dict): list_to_iterate = list(container.keys())
    else: raise Exception(NotImplementedError)

    ## Iterate and stack first index or key.
    for idx in list_to_iterate:
        if is_container(container[idx]): current_indices.append([idx])
        elif leaf_include_condition(container[idx]): final_indices.append([idx])
    return final_indices + get_paths_to_leaves_rec(container, current_indices, leaf_include_condition = leaf_include_condition)

def is_container(something):
    """Check whether something is container object.
    
    Examples
    --------
    print(is_container(["hi"]))
        >>> True
    """

    if isinstance(something, (list, tuple, dict, type(range(7)))):
        return True
    else:
        return False

class Grid():
    """Grid object used in get_list_of_grids function for grid search.
    
    Put this where you want to search grids of it.
    """

    def __init__(self, *list_of_components):
        self.list_of_components = list_of_components

class GridSearch():
    """Class for grid search with the different combination of parameters (grids)."""

    def __init__(self, container, key_of_name = None):
        """Init.
        
        Parameters
        ----------
        container : dict or list or tuple
            Container which will be shallow copied for grid search. Container is expected to contain 'Grid' object.
        key_of_name : index or hashable key
            The key/index to access the name of each grid container. The automatic number is added to the name."""

        self.container = container
        self.key_of_name = key_of_name
        self.list_of_paths_to_grids = get_paths_to_leaves(container, leaf_include_condition = lambda x: True if isinstance(x, Grid) else False) ## [[1, 'b', 3], [2, 0], [2, 1], [2, 2, 0], [2, 2, 1], [3], ...]
    
    def get_list_of_grids(self):
        """Grid Search.

        The Grid object can be located deep inside of nested list/dict.
        
        Examples
        --------
        test_grid_search_dict = dict(model_class = lambda x: True, iters = Grid(10, 20), model_structure = [["Dense", {"units": Grid(100, 200), "activation": Grid("tanh")}], Grid("hi", "bye")], name = "my model")\n
        for container_init in get_list_of_grids(container = test_grid_search_dict, key_of_name= "name"):
            print(f"{container_init}")
            >>> {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 10, 'model_structure': [['Dense', {'units': 100, 'activation': 'tanh'}], 'hi'], 'name': 'my model_0'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 20, 'model_structure': [['Dense', {'units': 100, 'activation': 'tanh'}], 'hi'], 'name': 'my model_1'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 10, 'model_structure': [['Dense', {'units': 100, 'activation': 'tanh'}], 'bye'], 'name': 'my model_2'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 20, 'model_structure': [['Dense', {'units': 100, 'activation': 'tanh'}], 'bye'], 'name': 'my model_3'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 10, 'model_structure': [['Dense', {'units': 200, 'activation': 'tanh'}], 'hi'], 'name': 'my model_4'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 20, 'model_structure': [['Dense', {'units': 200, 'activation': 'tanh'}], 'hi'], 'name': 'my model_5'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 10, 'model_structure': [['Dense', {'units': 200, 'activation': 'tanh'}], 'bye'], 'name': 'my model_6'}
            {'model_class': <function <lambda> at 0x1033d3040>, 'iters': 20, 'model_structure': [['Dense', {'units': 200, 'activation': 'tanh'}], 'bye'], 'name': 'my model_7'}
            
        """

        list_of_containers_init = [deepcopy(self.container)] ## Final result: realized grids.

        ## For each path to Grid object.
        for path in self.list_of_paths_to_grids: ## e.g. path = [1, 'b', 3].
            grid = access_with_list_of_keys_or_indices(self.container, path) ## This is grid object.
            if len(grid.list_of_components) == 0: ## CASE Grid(): Set default grid value None.
                for container_init in list_of_containers_init: ## e.g. container_init = dict(model_class = lambda x: True, iters = Grid(10, 20), model_structure = [["Dense", {"units": Grid(100, 200), "activation": Grid("tanh")}], Grid("hi", "bye")], name = "my model")
                    access_with_list_of_keys_or_indices(container_init, path[:-1])[path[-1]] = None ## Edit empty grid as None.
            else: ## Not empty grid.
                for container_init in list_of_containers_init: ## Apply first component of grid for efficient computation.
                    access_with_list_of_keys_or_indices(container_init, path[:-1])[path[-1]] = grid.list_of_components[0]
                
                ## Apply remaining components.
                list_of_containers_init_for_this_grid = [] ## Added containers because of current grid.
                for component_idx in range(1, len(grid.list_of_components)): ## First index is already applied above.
                    list_of_containers_init_copy = deepcopy(list_of_containers_init) ## All of containers which will be edited by the components of current grid.
                    for container_init in list_of_containers_init_copy: ## Edit the container with realized component.
                        access_with_list_of_keys_or_indices(container_init, path[:-1])[path[-1]] = grid.list_of_components[component_idx]
                    list_of_containers_init_for_this_grid = list_of_containers_init_for_this_grid + list_of_containers_init_copy ## Adds this realized component of current grid.
                list_of_containers_init = list_of_containers_init + list_of_containers_init_for_this_grid ## Adds all the realized components of current grid.
        
        ## Append numbering for each instance.
        if self.key_of_name is not None:
            assert(isinstance(self.container[self.key_of_name], str))
            for container_init, idx in zip(list_of_containers_init, range(len(list_of_containers_init))):
                if idx > 0:
                    container_init[self.key_of_name] = container_init[self.key_of_name] + f"_{idx}"
        return list_of_containers_init

def whether_lists_have_same_values(*lists, any_or_all = "all"):
    """Compare the values of lists.
    
    Examples
    --------
    test_list_1 = [[1, 2], [2, 1], [3, 4, 5], [], [3, 4, 5], [6]]
    print(whether_lists_have_same_values(*test_list_1, any_or_all = "all"))
        >>> False
    print(whether_lists_have_same_values(*test_list_1, any_or_all = "any"))
        >>> True
    test_list_2 = [[6, "hi"], [6, "hi"], [6, "hi"]]
    print(whether_lists_have_same_values(*test_list_2, any_or_all = "all"))
        >>> True
    print(whether_lists_have_same_values(*test_list_2, any_or_all = "any"))
        >>> True
    """
    
    base_idx = 1 ## pivot.
    assert(len(lists) > 1)
    for i in range(len(lists)): ## lists[i] is pivot list.
        for j in range(i + 1, len(lists)): ## Compare with j-th list.
            if lists[i] != lists[j]:
                if any_or_all == "all": return False
            else:
                if any_or_all == "any": return True
        if any_or_all == "all": return True
    return False ## For the case any_or_all == "any".

def get_items_counts_dict_from_container(container, access_to_item_funct = lambda x: x):
    """
    
    Examples
    --------
    test_dicts = [{"predictor label": np.array([[0., 1.]])}, {"predictor label": np.array([[0., 1.]])}, {"predictor label": np.array([[1., 0.]])}, {"predictor label": np.array([[0., 1.]])}]
    print(get_items_counts_dict_from_container([0, 2, 3], access_to_item_funct = lambda x: test_dicts[x]["predictor label"]))
        >>> {((0.0, 1.0),): 2, ((1.0, 0.0),): 1}
    """
    counts_dict = {}
    for item in container:
        sub_item = access_to_item_funct(item)
        if isinstance(sub_item, np.ndarray):
            sub_item = tuple(map(tuple, sub_item))
        if sub_item in counts_dict.keys():
            counts_dict[sub_item] += 1
        else:
            counts_dict[sub_item] = 1
    return counts_dict

def get_max_with_accessor(container, accessor = lambda x: x):
    """

    Examples
    --------
    test_dict = {0: 3, 1: 2, 5: 7, "a": 4}
    print(get_max_with_accessor(test_dict, accessor = lambda x: test_dict[x]))
        >>> 7
    """
    current_max = accessor(next(iter(container)))
    for item in container:
        if current_max < accessor(item):
            current_max = accessor(item)
    return current_max

if __name__ == "__main__":
    pass
    test_dicts = [{"predictor label": np.array([[0., 1.]])}, {"predictor label": np.array([[0., 1.]])}, {"predictor label": np.array([[1., 0.]])}, {"predictor label": np.array([[0., 1.]])}]
    print(get_items_counts_dict_from_container([0, 2, 3], access_to_item_funct = lambda x: test_dicts[x]["predictor label"]))

