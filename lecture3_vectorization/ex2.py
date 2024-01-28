import numpy as np
import time
import mlx.core as mx


def add_python(size):
    a = list(range(size))
    b = list(range(size))

    start = time.time()
    for i in range(size):
        a[i] += b[i]
    end = time.time()

    return end - start


def add_numpy(size):
    na = np.random.randint(1, 1000, size)
    nb = np.random.randint(1, 1000, size)

    start = time.time()
    na += nb
    end = time.time()

    return end - start


def add_mlx(size):
    ma = mx.array(np.random.randint(1, 1000, size))
    mb = mx.array(np.random.randint(1, 1000, size))

    start = time.time()
    mx.add(ma, mb, stream=mx.cpu)
    end = time.time()

    return end - start


def benchmark(func, size, num_tests=5):
    results = []

    for i in range(num_tests):
        start = time.time()
        func(size)
        end = time.time()
        results.append(end - start)

    reported_results = results.copy()
    reported_results.remove(max(results))
    reported_results.remove(min(results))

    return np.prod(reported_results) ** (1.0 / len(reported_results))


def main(args):
    SIZE = 6400000
    add_python_result = benchmark(add_python, SIZE)
    add_numpy_result = benchmark(add_numpy, SIZE)

    print(f"add_python={add_python_result} s")
    print(f"add_numpy={add_numpy_result} s")
    print(f"speedup={add_python_result/add_numpy_result}")

    f = open("ex2.txt", "w")
    f.write(f"add_python={add_python_result} s\n")
    f.write(f"add_numpy={add_numpy_result} s\n")
    f.write(f"speedup={add_python_result/add_numpy_result}\n")

    if args.extra:
        add_mlx_result = benchmark(add_mlx, SIZE)
        print(f"add_mlx={add_mlx_result} s")
        print(f"speedup={add_python_result/add_mlx_result}")

        f.write(f"add_mlx={add_mlx_result} s\n")
        f.write(f"speedup={add_python_result/add_mlx_result}\n")

    f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark addition")
    parser.add_argument("--extra", action="store_true", help="Run extra benchmarks")

    args = parser.parse_args()

    main(args)
