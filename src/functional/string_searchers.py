"""
    Implements any functions which look for strings.

    Possibly also vectorize these functions if possible
      (although numpy and string ehhh)
"""

def find_all_indecies_subarray(subarray, array):
    """
        Finds all indecies where `subarray` is included in `array`
    :param arr1:
    :param arr2:
    :return:
    """
    window_size = len(subarray)
    subarray_idx = [x for x in range(len(array)) if array[x:x + window_size] == subarray]
    return subarray_idx