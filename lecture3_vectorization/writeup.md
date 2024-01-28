# Lecture 3 Exercise - SIMD & Vectorization

## Exercise 1

To measure the speedup of the vectorized code in C, I experimented with 5 run on each version of the code, Remove minimum and maximum result of that 5 run and calculate the geometric mean of the remaining 3 run. The result is shown in the table below.

[Code](./ex1.c) | [Result](./ex1.txt)

| Version          | Time (s) | Speedup |
| ---------------- | -------- | ------- |
| Scalar (No AVX)  | 0.056684 | 1.00    |
| Vector (w/ AVX2) | 0.009861 | 5.75    |

## Exercise 2

To measure the speedup of the vectorized code in Python, I experimented with 5 run on each version of the code, Remove minimum and maximum result of that 5 run and calculate the geometric mean of the remaining 3 run. The result is shown in the table below.

[Code](./ex2.py) | [Result](./ex2.txt)

| Version         | Time (s) | Speedup |
| --------------- | -------- | ------- |
| Scalar (Python) | 0.556130 | 1.00    |
| Vector (Numpy)  | 0.049630 | 11.21   |

## Exercise 3

Vectorization may not be beneficial in these situations:

- Some compilers might not be able to vectorize the code automatically, so we have to explicitly write the vectorized code with knowledge of the computer architecture.
- Some programs are not suitable for vectorization, for example, programs that have a lot of conditions or branches, or programs that have a lot of data dependencies, vectorization becomes complex.
