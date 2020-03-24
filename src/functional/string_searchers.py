"""
    Implements any functions which look for strings.

    Possibly also vectorize these functions if possible
      (although numpy and string ehhh)
"""

def find_all_indecies_subarray(subarray, array, fn_stem=None):
    """
        Finds all indecies where `subarray` is included in `array`
    :param arr1:
    :param arr2:
    :return:
    """
    assert isinstance(subarray, list), subarray
    assert isinstance(array, list), array

    # print("Got following inputs")
    # print(subarray)
    # print(array)

    # Should perhaps use a stemmer ...
    if fn_stem is not None:
        array = [fn_stem(x) for x in array]
        subarray = [fn_stem(x) for x in subarray]
    # print("After stemmer")
    # print(subarray)
    # print(array)

    window_size = len(subarray)
    subarray_idx = [x for x in range(len(array)) if array[x:x + window_size] == subarray]

    # print("What is returned...")
    # print(window_size)
    # print(subarray_idx)

    return subarray_idx
