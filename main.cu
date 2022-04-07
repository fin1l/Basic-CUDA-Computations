// GPU CUDA Computations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#include <string.h>
#include <chrono>
#include "CUDA_Header.cuh"
#include "cuda_runtime.h"
#include "Adding.h"

int main(void)
{
    const int N = 1 << 15; //1024 operations is the when they are approximately equal on my machine (1 << 10)

    float* x;
    float* y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    int iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    // Run kernel on 1M elements on the CPU
    int blockS = 512;
    int blockN = (N + blockS - 1) / blockS;
    for (int ite = 0; ite < iterations; ite++)
    {
        addKer<<<blockN, blockS>>> (N, x, y);
    }
    //addKerCpu(N, x, y);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "CUDA Multithreaded: " << duration.count()/iterations << "ns\n";
    cudaDeviceSynchronize();


    // Free memory
    cudaFree(x);
    cudaFree(y);

    float xn[N];
    float yn[N];
    
    for (int i = 0; i < N; i++) {
        xn[i] = 1.0f;
        yn[i] = 2.0f;
    }

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int iters = 0; iters < iterations; iters++)
    {
        for (int i = 0; i < N; i++) {
            CPU(xn, yn, i);
        }
    }
    
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop2 - start2);
    std::cout << "Single CPU Thread: " << duration2.count() / iterations << "ns";
    std::string c;
    std::cin >> c;

    return 0;
}