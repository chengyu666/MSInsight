// 把走时纠正值先存入shared memory

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

inline int getLinearIndex2d(int i, int j, int cols)
{
    return i * cols + j;
}

inline int getLinearIndex3d(int i, int j, int k, int cols, int rows)
{
    return i * cols * rows + j * cols + k;
}

/**
 * @brief 用于在CUDA中执行栈叠操作的核函数。计算一个网格点、机制、采样点下，所有站数据的栈叠结果。
 *
 * @param data 输入数据数组，大小为 n_sta * n_samples，存储每个站的数据。
 * @param tt_samples 每个站点的时间偏移数组，大小为 n_sta * n_grid，存储每个站在每个网格点的时间偏移。
 * @param result 输出结果数组，大小为 n_grid * n_samples，用于存储栈叠后的结果。
 * @param n_sta 站点数量。
 * @param n_samples 每个站点的采样点数量。
 * @param n_grid 网格点数量。
 * @param max_i_sample 每个kernel最大的采样点索引。
 * @param i_sample_offset 当前kernel的采样点偏移量。
 */
__global__ void stackKernel(const float *data, const int *tt_samples,
                            float *result, int n_sta, int n_samples, int n_grid,
                            int max_i_sample, int i_sample_offset)
{
    int i_sample = threadIdx.x;                    // 当前kernel内的sample索引
    int sample_index = i_sample + i_sample_offset; // 整体数据中采样点索引
    // if (sample_index >= max_i_sample)
    //     return;
    int64_t i_grid = blockIdx.x;

    // 加载tt_samples到共享内存
    extern __shared__ int shared_tt_samples[];
    if (threadIdx.x == 0)
    {
        for (int i_sta = 0; i_sta < n_sta; ++i_sta)
        {
            shared_tt_samples[i_sta] = tt_samples[i_sta * n_grid + i_grid];
        }
    }
    __syncthreads(); // 确保所有线程都加载完毕

    float sum = 0.0f;
    float shifted_data = 0.0f;
    int offset = 0;
    for (int i_sta = 0; i_sta < n_sta; ++i_sta)
    {
        // 遍历每个站点，计算所有站点的栈叠结果
        offset = shared_tt_samples[i_sta];
        shifted_data = data[i_sta * n_samples + (sample_index + offset)];
        sum += shifted_data;
    }
    sum /= n_sta; // 对所有站叠加结果求平均
    result[i_grid * n_samples + sample_index] = sum;
}

/**
 * @brief 调用CUDA核函数进行栈叠操作的函数。计算所有网格点、机制、采样点下，栈叠结果。
 *
 * @param data 输入数据数组，大小为 n_sta * n_samples，存储每个站的数据。
 * @param tt_samples 每个站点的时间偏移数组，大小为 n_sta * n_grid，存储每个站在每个网格点的时间偏移。
 * @param n_sta 站点数量。
 * @param n_samples 每个站点的采样点数量。
 * @param n_grid 网格点数量。
 * @param result 输出结果数组，大小为 n_grid * n_samples，用于存储栈叠后的结果。
 */
__declspec(dllexport) int stackCUDA(const float *data, const int *tt_samples,
                                    int n_sta, int n_samples, int n_grid,
                                    float *result)
{
    std::cout << "[SSA.cu] running stackCUDA " << std::endl;

    // Allocate device memory
    float *d_data, *d_result;
    int *d_tt_samples;
    std::cout << "[SSA.cu] n_sta: " << n_sta << " n_samples: " << n_samples << " n_grid: " << n_grid << std::endl;

    // print memory usage
    std::cout << "[SSA.cu] use GPU memory: " << n_sta * n_samples * sizeof(float) + n_sta * n_grid * sizeof(int) + n_grid * n_samples * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] data size: " << n_sta * n_samples * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] tt_samples size: " << n_sta * n_grid * sizeof(int) << std::endl;
    std::cout << "[SSA.cu] result size: " << n_grid * n_samples * sizeof(float) << std::endl;

    if (cudaMalloc(&d_data, n_sta * n_samples * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_data" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_tt_samples, n_sta * n_grid * sizeof(int)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_tt_samples" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_result, n_grid * n_samples * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_result" << std::endl;
        return -1;
    }

    // Copy data to device
    cudaMemcpy(d_data, data, n_sta * n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tt_samples, tt_samples, n_sta * n_grid * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, n_grid * n_samples * sizeof(float));

    size_t shared_mem_size = n_sta * sizeof(int);

    // 计算每次能处理的数据长度
    const int *p_max_tt = std::max_element(tt_samples, tt_samples + n_sta * n_grid); // 找到所有走时中的最大值
    int max_i_sample = n_samples - *p_max_tt; // 整个计算最大的i_samples值，超过的会有部分数据下标超范围，计算没意义
    int sample_offset = 0;                    // i_samples的偏移量，从0开始
    std::cout << "[SSA.cu] max_tt: " << *p_max_tt << " max_i_sample : " << max_i_sample << std::endl;
    int sample_remaining = max_i_sample; // 剩余的采样点数
    // 所有kernel执行时间的总和
    std::chrono::duration<double, std::milli> total_duration(0.0);

    // 循环执行，处理所有数据
    while (sample_remaining > 0)
    {
        // Define grid and block dimensions
        dim3 gridDim(n_grid);
        int n_threads = sample_remaining > 1024 ? 1024 : sample_remaining; // 每个Block中最多可以包含1024个线程，这里的n_threads不能超过1024
        dim3 blockDim(n_threads);

        std::cout << "[SSA.cu] sample_remaining: " << sample_remaining << " n_threads: " << n_threads << std::endl;

        auto start = std::chrono::high_resolution_clock::now(); // 开始时间

        // Launch kernel
        stackKernel<<<gridDim, blockDim, shared_mem_size>>>(d_data, d_tt_samples, d_result,
                                                            n_sta, n_samples, n_grid, max_i_sample, sample_offset);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();             // 结束时间
        std::chrono::duration<double, std::milli> duration = end - start; // 计算时间差
        std::cout << "[SSA.cu] kernel execution time: " << duration.count() << " ms" << std::endl;
        total_duration += duration; // 累加时间

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        sample_remaining -= 1024; // 每次处理最多max_i_sample个采样点
        sample_offset += 1024;    // 偏移量增加max_i_sample
    }
    std::cout << "[SSA.cu] total kernel execution time: " << total_duration.count() << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(result, d_result, n_grid * n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_tt_samples);
    cudaFree(d_result);
    return 0;
}

void testStack()
{
    std::cout << "CUDA Version: " << CUDART_VERSION << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << ", sizeof(int): " << sizeof(int) << std::endl;
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU数量: " << count << std::endl;
    std::cout << "每个Block最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "每个Block最大线程维度: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "每个Grid最大线程数: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    std::cout << "每个Block最大共享内存: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "总显存: " << prop.totalGlobalMem << std::endl;
    // Example usage
    int n_sta = 40;        // Number of stations
    int n_samples = 7501; // Number of samples per station
    int n_grid = 48000;    // Number of grid points

    float *data = new float[n_sta * n_samples];
    int *tt_samples = new int[n_sta * n_grid];
    float *result = new float[n_grid * n_samples]{0};

    // Initialize example data
    for (int i = 0; i < n_sta * n_samples; ++i)
        data[i] = 1.0f;

    for (int i = 0; i < n_sta * n_grid; ++i)
    {
        tt_samples[i] = rand() % 100 + 1;
        // std::cout << tt_samples[i] << " ";
    }
    // std::cout << std::endl;
    std::cout << "tt_samples:";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << tt_samples[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Start stackCUDA" << std::endl;
    // Call the stackCUDA function
    stackCUDA(data, tt_samples, n_sta, n_samples, n_grid, result);
    std::cout << "End stackCUDA" << std::endl;
    // Print a portion of the result
    std::cout << "result:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 1; i < 11; ++i)
    {
        std::cout << result[n_grid * n_samples-i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] data;
    delete[] tt_samples;
    delete[] result;
}

int main()
{
    testStack();
    return 0;
}