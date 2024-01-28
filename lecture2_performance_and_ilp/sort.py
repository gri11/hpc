import argparse
import time
import numpy as np
import numba
import pickle
import pandas as pd
import mlx.core as mx


def sort_list(ls):
    return sorted(ls)


def np_sort_list(ls):
    return np.sort(ls)


def mlx_sort_list(ls):
    # return mx.sort(mx.array(ls)[:2_000_000])
    return mx.sort(mx.array(ls), stream=mx.cpu)


@numba.njit
def numba_quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return numba_quicksort(less) + [pivot] + numba_quicksort(greater)


def extract_data(input_file):
    with open(input_file) as f:
        key2value = {
            key: float(value)
            for key, value in [line.split(":") for line in f.read().splitlines()]
        }

    value2key = {value: key for key, value in key2value.items()}
    unsorted_list = list(value2key.keys())

    with open("unsorted_list.pkl", "wb") as f:
        pickle.dump(unsorted_list, f)

    return unsorted_list


def benchmark(unsorted_list, time_dict={}):
    if "builtin_time" not in time_dict:
        time_dict["builtin_time"] = []
    start_time = time.time()
    sort_list(unsorted_list)
    time_dict["builtin_time"] = time_dict["builtin_time"] + [time.time() - start_time]
    print(f"{time_dict['builtin_time']} s")

    if "numpy_time" not in time_dict:
        time_dict["numpy_time"] = []
    start_time = time.time()
    np_sort_list(unsorted_list)
    time_dict["numpy_time"] = time_dict["numpy_time"] + [time.time() - start_time]
    print(f"{time_dict['numpy_time']} s")

    if "mlx_time" not in time_dict:
        time_dict["mlx_time"] = []
    start_time = time.time()
    mlx_sort_list(unsorted_list)
    time_dict["mlx_time"] = time_dict["mlx_time"] + [time.time() - start_time]
    print(f"{time_dict['mlx_time']} s")

    return time_dict


def main():
    parser = argparse.ArgumentParser(description="Sort a list of numbers")
    parser.add_argument("--input", help="Input file")
    args = parser.parse_args()

    # extract_data(args.input)

    unsorted_list = pickle.load(open("unsorted_list.pkl", "rb"))

    time_dict = {}

    for i in range(5):
        time_dict = benchmark(unsorted_list, time_dict)

    print(pd.DataFrame(time_dict).describe())


if __name__ == "__main__":
    main()
