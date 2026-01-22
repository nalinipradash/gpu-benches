#include "../gpu-error.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <cmath>

using namespace std::chrono;

template <typename T> __global__ void initKernel(T *data, size_t data_len) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x) {
    data[idx] = idx;
  }
}

// Runtime version of testfun where N is passed as a parameter
template <typename T, int M, int BLOCKSIZE>
__global__ void testfun_runtime(T *const __restrict__ dA, T *const __restrict__ dB,
                                T *dC, int N) {
  T *sA = dA + threadIdx.x + blockIdx.x * BLOCKSIZE * M;
  T *sB = dB + threadIdx.x + blockIdx.x * BLOCKSIZE * M;

  T sum = 0;

  for (int i = 0; i < M; i += 2) {
    T a = sA[i * BLOCKSIZE];
    T b = sB[i * BLOCKSIZE];
    T v = a - b;
    T a2 = sA[(i + 1) * BLOCKSIZE];
    T b2 = sB[(i + 1) * BLOCKSIZE];
    T v2 = a2 - b2;
    for (int j = 0; j < N; j++) {
      v = v * a - b;
      v2 = v2 * a - b;
    }
    sum += v + v2;
  }
  if (threadIdx.x == 0)
    dC[blockIdx.x] = sum;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }

  double target_ai = atof(argv[1]);
  double duration_sec = atof(argv[2]);

  if (target_ai <= 0 || duration_sec <= 0) {
    return 1;
  }

  typedef float dtype;
  const int M = 4000;
  const int BLOCKSIZE = 256;

  // Calculate N from algorithmic intensity
  // AI = (2.0 + N * 2.0) / (2.0 * sizeof(dtype))
  // AI = (2.0 + N * 2.0) / 8.0
  // AI = (1.0 + N) / 4.0
  // 4 * AI = 1.0 + N
  // N = 4 * AI - 1
  int N = (int)round(4.0 * target_ai - 1.0);
  if (N < 0) N = 0;

  int nDevices;
  GPU_ERROR(cudaGetDeviceCount(&nDevices));

#pragma omp parallel num_threads(nDevices)
  {
    GPU_ERROR(cudaSetDevice(omp_get_thread_num()));
#pragma omp barrier
    int deviceId;
    GPU_ERROR(cudaGetDevice(&deviceId));
    cudaDeviceProp prop;
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    int numBlocks;

    // Use reasonable default for block count
    numBlocks = 8;
    int blockCount = prop.multiProcessorCount * numBlocks;

    size_t data_len = (size_t)blockCount * BLOCKSIZE * M;
    dtype *dA = NULL;
    dtype *dB = NULL;
    dtype *dC = NULL;

    GPU_ERROR(cudaMalloc(&dA, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dB, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dC, data_len * sizeof(dtype)));
#pragma omp barrier
    initKernel<<<blockCount, 256>>>(dA, data_len);
    initKernel<<<blockCount, 256>>>(dB, data_len);
    initKernel<<<blockCount, 256>>>(dC, data_len);
    GPU_ERROR(cudaDeviceSynchronize());

#pragma omp barrier

    auto start_time = steady_clock::now();
    auto end_time = start_time + duration<double>(duration_sec);

    // Run kernels continuously until time expires
    while (steady_clock::now() < end_time) {
      testfun_runtime<dtype, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC, N);
    }

    GPU_ERROR(cudaDeviceSynchronize());
    GPU_ERROR(cudaGetLastError());
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
    GPU_ERROR(cudaFree(dC));
  }
  return 0;
}
