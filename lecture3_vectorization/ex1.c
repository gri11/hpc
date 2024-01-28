#include <stdio.h>
#include <time.h>
#include <math.h>
#include <smmintrin.h>
#include <immintrin.h>

// No AVX
void add(int size, int *a, int *b)
{
    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }
}

// with AVX2
void add_avx(int size, int *a, int *b)
{
    int i = 0;
    for (; i < size; i += 8)
    {
        // load 256-bit chunks of each array
        __m256i av = _mm256_loadu_si256((__m256i *)&a[i]);
        __m256i bv = _mm256_loadu_si256((__m256i *)&b[i]);
        // add each pair of 32-bit integers in chunks
        av = _mm256_add_epi32(av, bv);

        // store 256-bit chunk to a
        _mm256_storeu_si256((__m256i *)&a[i], av);
    }
    // clean up
    for (; i < size; i++)
    {
        a[i] += b[i];
    }
}

float benchmark(void (*f)(int, int *, int *), int num_test)
{
    float results[num_test], min = 100., max = 0.;

    const int SIZE = 1e6;
    int a[SIZE], b[SIZE];

    for (int i = 0; i < num_test; i++)
    {
        float startTime = (float)clock() / CLOCKS_PER_SEC;
        (*f)(SIZE, a, b);
        float endTime = (float)clock() / CLOCKS_PER_SEC;
        float timeElapsed = endTime - startTime;
        results[i] = timeElapsed;

        if (timeElapsed < min)
            min = timeElapsed;

        if (timeElapsed > max)
            max = timeElapsed;
    }

    float result = 1;
    for (int i = 0; i < num_test; i++)
    {
        if (results[i] == min)
            continue;
        if (results[i] == max)
            continue;
        result *= results[i];
    }

    return pow(result, 1. / (num_test - 2));
}

int main()
{
    float add_result = benchmark(add, 5);
    float add_avx_result = benchmark(add_avx, 5);
    printf("add benchmark=%f s\n", add_result);
    printf("add_avx benchmark=%f s\n", add_avx_result);
    printf("speedup=%f\n", add_result / add_avx_result);

    // Write to `ex1.txt`
    FILE *fptr;

    fptr = fopen("ex1.txt", "w");

    fprintf(fptr, "add benchmark=%f s\n", add_result);
    fprintf(fptr, "add_avx benchmark=%f s\n", add_avx_result);
    fprintf(fptr, "speedup=%f\n", add_result / add_avx_result);

    fclose(fptr);

    return 0;
}