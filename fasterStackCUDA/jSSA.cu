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
 * @param intensity 强度数组，大小为 n_fm * n_grid * n_sta，存储每个机制、网格点和站的强度值。
 * @param result 输出结果数组，大小为 n_grid * n_fm * n_samples，用于存储栈叠后的结果。
 * @param n_sta 站点数量。
 * @param n_samples 每个站点的采样点数量。
 * @param n_grid 网格点数量。
 * @param n_fm 机制数量。
 */
__global__ void stackKernel(const float *data, const int *tt_samples, const float *intensity,
                            float *result, int n_sta, int n_samples, int n_grid, int n_fm)
{
    int i_grid = blockIdx.x;
    int i_fm = blockIdx.y;
    int i_sample = threadIdx.x;

    if (i_sample >= n_samples)
        return;

    float sum = 0.0f;
    float shifted_data = 0.0f;
    for (int i_sta = 0; i_sta < n_sta; ++i_sta)
    {
        // 遍历每个站点，计算所有站点的栈叠结果
        int offset = tt_samples[i_sta * n_grid + i_grid];
        if (i_sample + offset < n_samples) // 保证不越界
        {
            shifted_data = data[i_sta * n_samples + (i_sample + offset)];
            shifted_data *= intensity[i_fm * n_grid * n_sta + i_grid * n_sta + i_sta];
            sum += shifted_data;
        }
    }
    sum /= n_sta; // 对所有站叠加结果求平均
    result[i_grid * n_fm * n_samples + i_fm * n_samples + i_sample] = sum;
    // 安全地对共享变量执行加法操作
    // atomicAdd(&result[i_grid * n_fm * n_samples + i_fm * n_samples + i_sample], sum);
    // atomicAdd(&result[getLinearIndex3d(i_grid, i_fm, i_sample, n_fm, n_samples)], sum);
}

/**
 * @brief 用于在CUDA中执行联合栈叠操作的函数。计算一个网格点、机制、采样点下，所有站数据的栈叠结果。
 *
 * @param data 输入数据数组，大小为 n_sta * n_samples，存储每个站的数据。
 * @param tt_samples 每个站点的时间偏移数组，大小为 n_sta * n_grid，存储每个站在每个网格点的时间偏移。
 * @param intensity 强度数组，大小为 n_fm * n_grid * n_sta，存储每个机制、网格点和站的强度值。
 * @param n_sta 站点数量。
 * @param n_samples 每个站点的采样点数量。
 * @param n_grid 网格点数量。
 * @param n_fm 机制数量。
 * @param result 输出结果数组，大小为 n_grid * n_fm * n_samples，用于存储栈叠后的结果。
 */
__declspec(dllexport) int stackCUDA(const float *data, const int *tt_samples, const float *intensity,
                                    int n_sta, int n_samples, int n_grid, int n_fm,
                                    float *result)
{
    std::cout << "[SSA.cu] running stackCUDA " << std::endl;
    // Allocate device memory
    float *d_data, *d_intensity, *d_result;
    int *d_tt_samples;
    std::cout << "[SSA.cu] n_sta: " << n_sta << " n_samples: " << n_samples << " n_grid: " << n_grid << " n_fm: " << n_fm << std::endl;

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
    if (cudaMalloc(&d_intensity, n_fm * n_grid * n_sta * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_intensity" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_result, n_grid * n_fm * n_samples * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_result" << std::endl;
        return -1;
    }

    std::cout << "[SSA.cu] use GPU memory: " << n_sta * n_samples * sizeof(float) + n_sta * n_grid * sizeof(int) + n_fm * n_grid * n_sta * sizeof(float) + n_grid * n_fm * n_samples * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] data size: " << n_sta * n_samples * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] tt_samples size: " << n_sta * n_grid * sizeof(int) << std::endl;
    std::cout << "[SSA.cu] intensity size: " << n_fm * n_grid * n_sta * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] result size: " << n_grid * n_fm * n_samples * sizeof(float) << std::endl;

    // -------------------------- 计时修改核心 --------------------------
    auto start = std::chrono::high_resolution_clock::now(); // 开始时间：移到拷贝之前，包含所有操作

    // Copy data to device（保持不变）
    cudaMemcpy(d_data, data, n_sta * n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tt_samples, tt_samples, n_sta * n_grid * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intensity, intensity, n_fm * n_grid * n_sta * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, n_grid * n_fm * n_samples * sizeof(float));

    // Define grid and block dimensions（保持不变）
    dim3 gridDim(n_grid, n_fm);
    dim3 blockDim(n_samples); // 每个Block中最多可以包含1024个线程，这里的n_samples不能超过1024

    // Launch kernel（保持不变）
    stackKernel<<<gridDim, blockDim>>>(d_data, d_tt_samples, d_intensity, d_result,
                                       n_sta, n_samples, n_grid, n_fm);
    cudaDeviceSynchronize(); // 等待核函数完成

    // Copy result back to host（保持不变，纳入计时范围）
    cudaMemcpy(result, d_result, n_grid * n_fm * n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();             // 结束时间：移到结果拷贝之后
    std::chrono::duration<double, std::milli> duration = end - start; // 计算总耗时
    // 打印改为“总执行时间”，与异步版本格式一致，方便对比
    std::cout << "[SSA.cu] total execution time (copy + kernel + copy back): " << duration.count() << " ms" << std::endl;

    // -------------------------- 其余部分保持不变 --------------------------
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        // 出错时释放内存，避免泄漏
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_tt_samples);
    cudaFree(d_intensity);
    cudaFree(d_result);
    return 0;
}

#include <cmath>
#include <array>
#include <vector>
#define M_PI 3.14159265358979323846

/**
 * @brief Converts fault parameters (strike, dip, rake) to a moment tensor.
 *
 * @param strike Strike angle in degrees.
 * @param dip Dip angle in degrees.
 * @param rake Rake angle in degrees.
 * @param moment_tensor Output moment tensor as a 3x3 matrix.
 */
__device__ void faultParametersToMomentTensor(float strike, float dip, float rake, float moment_tensor[3][3])
{
    // Convert angles from degrees to radians
    strike = strike * M_PI / 180.0f;
    dip = dip * M_PI / 180.0f;
    rake = rake * M_PI / 180.0f;

    // Calculate moment tensor components
    moment_tensor[0][0] = -std::sin(dip) * std::cos(rake) * std::sin(2 * strike) -
                          std::sin(2 * dip) * std::sin(rake) * std::pow(std::sin(strike), 2);
    moment_tensor[1][1] = std::sin(dip) * std::cos(rake) * std::sin(2 * strike) -
                          std::sin(2 * dip) * std::sin(rake) * std::pow(std::cos(strike), 2);
    moment_tensor[0][1] = std::sin(dip) * std::cos(rake) * std::cos(2 * strike) +
                          0.5f * std::sin(2 * dip) * std::sin(rake) * std::sin(2 * strike);
    moment_tensor[1][0] = moment_tensor[0][1];
    moment_tensor[0][2] = -std::cos(dip) * std::cos(rake) * std::cos(strike) -
                          std::cos(2 * dip) * std::sin(rake) * std::sin(strike);
    moment_tensor[2][0] = moment_tensor[0][2];
    moment_tensor[1][2] = -std::cos(dip) * std::cos(rake) * std::sin(strike) +
                          std::cos(2 * dip) * std::sin(rake) * std::cos(strike);
    moment_tensor[2][1] = moment_tensor[1][2];
    moment_tensor[2][2] = std::sin(2 * dip) * std::sin(rake);
}

/**
 * @brief Calculates the radiation intensity for a given vector and moment tensor.
 *        The vector direction should be from the source to the receiver (north, east, down components).
 * @param moment_tensor Moment tensor as a 3x3 matrix.
 * @param vector Direction vector from the source to the receiver (north, east, down components).
 *                The vector should be normalized before passing it to this function.
 * @return float Radiation intensity.
 */
__device__ void calculateRadiationIntensity(const float moment_tensor[3][3], const float vector[3], float &intensity)
{
    // Normalize the vector
    float norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    float normalized_vector[3] = {vector[0] / norm, vector[1] / norm, vector[2] / norm};

    // Calculate the radiation intensity
    // float intensity = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            intensity += normalized_vector[i] * moment_tensor[i][j] * normalized_vector[j];
        }
    }
}

typedef struct
{
    float GridSpacingX;
    float GridSpacingZ;
    int SearchSizeX;
    int SearchSizeY;
    int SearchSizeZ;
    float SearchOriginX;
    float SearchOriginY;
    float SearchOriginZ;
} GridConfig;

__global__ void intensityKernel(const float *fm_grid, const GridConfig *conf, const float *stations,
                                float *result, int n_sta, int n_fm, int n_grid, float a)
{
    // 计算每个网格点、机制下，所有站数据的强度值
    int i_grid = blockIdx.x; // 网格点索引
    int i_fm = blockIdx.y;   // 机制索引
    int i_sta = threadIdx.x; // 站点索引

    // if (i_sta >= n_sta)
    //     return;

    // 计算网格点坐标
    // 注意：GridSpacingX 和 GridSpacingY 始终相同，因此直接使用 GridSpacingX
    float x = conf->SearchOriginX + (i_grid / (conf->SearchSizeY * conf->SearchSizeZ)) * conf->GridSpacingX;
    float y = conf->SearchOriginY + ((i_grid / conf->SearchSizeZ) % conf->SearchSizeY) * conf->GridSpacingX;
    float z = conf->SearchOriginZ + (i_grid % conf->SearchSizeZ) * conf->GridSpacingZ;

    // 计算距离
    float dx = x - stations[i_sta * 3];
    float dy = y - stations[i_sta * 3 + 1];
    float dz = z - stations[i_sta * 3 + 2];
    // 向量方向：北东下
    float vector[3] = {dy, dx, dz};
    float v_len = sqrt(dx * dx + dy * dy + dz * dz + 1e-6);

    // 计算强度值
    float moment_tensor[3][3];
    faultParametersToMomentTensor(fm_grid[i_fm * 3], fm_grid[i_fm * 3 + 1], fm_grid[i_fm * 3 + 2], moment_tensor);
    float *p = &result[i_fm * n_grid * n_sta + i_grid * n_sta + i_sta];
    calculateRadiationIntensity(moment_tensor, vector, *p);
    // *p = v_len;
    *p *= exp(-a * v_len);
    // result[i_fm * n_grid * n_sta + i_grid * n_sta + i_sta] = calculateRadiationIntensity(
    //                                                              moment_tensor, vector) *
    //                                                          exp(-a * v_len);
}

/**
 * @brief 用于在CUDA中执行强度计算的函数。计算每个网格点、机制下，所有站数据的强度值。
 *
 * @param fm_grid 输入数据数组，大小为 n_fm * 3，存储每个机制的参数。
 * @param conf 网格配置结构体，包含网格的参数。
 * @param stations 站点数组，大小为 n_sta * 3，存储每个站点的坐标。
 * @param n_sta 站点数量。
 * @param n_fm 机制数量。
 * @param a 距离衰减因子。
 * @param result 输出结果数组，大小为 n_fm * n_grid * n_sta，用于存储每个机制、网格点和站的强度值。
 */
__declspec(dllexport) int intensityCUDA(const float *fm_grid, GridConfig *conf, float *stations,
                                        int n_sta, int n_fm, float a,
                                        float *result)
{
    std::cout << "[SSA.cu] running intensityCUDA " << std::endl;
    // 网格点数量
    int n_grid = conf->SearchSizeX * conf->SearchSizeY * conf->SearchSizeZ;
    std::cout << "[SSA.cu] n_sta: " << n_sta << " n_fm: " << n_fm << " n_grid: " << n_grid << std::endl;

    // Allocate device memory
    float *d_fm_grid, *d_stations, *d_result;
    GridConfig *d_conf;
    if (cudaMalloc(&d_fm_grid, n_fm * 3 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_fm_grid" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_conf, sizeof(GridConfig)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_conf" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_stations, n_sta * 3 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_stations" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_result, n_fm * n_grid * n_sta * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_result" << std::endl;
        return -1;
    }
    std::cout << "[SSA.cu] use GPU memory: " << n_fm * 3 * sizeof(float) + sizeof(GridConfig) + n_sta * 3 * sizeof(float) + n_fm * n_grid * n_sta * sizeof(float) << std::endl;

    // Copy data to device
    cudaMemcpy(d_fm_grid, fm_grid, n_fm * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conf, conf, sizeof(GridConfig), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stations, stations, n_sta * 3 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(d_result, 0, n_fm * n_grid * n_sta * sizeof(float));

    // Define grid and block dimensions
    dim3 gridDim(n_grid, n_fm);
    dim3 blockDim(n_sta); // n_sta 不能超过1024
    // Launch kernel
    intensityKernel<<<gridDim, blockDim>>>(d_fm_grid, d_conf, d_stations, d_result,
                                           n_sta, n_fm, n_grid, a);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // Copy result back to host
    cudaMemcpy(result, d_result, n_fm * n_grid * n_sta * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_fm_grid);
    cudaFree(d_conf);
    cudaFree(d_stations);
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
    int n_sta = 80;      // Number of stations
    int n_samples = 512; // Number of samples per station
    int n_grid = 512;    // Number of grid points
    int n_fm = 576;      // Number of focal mechanisms

    float *data = new float[n_sta * n_samples];
    int *tt_samples = new int[n_sta * n_grid];
    float *intensity = new float[n_fm * n_grid * n_sta];
    float *result = new float[n_grid * n_fm * n_samples]{0};

    // Initialize example data
    for (int i = 0; i < n_sta * n_samples; ++i)
        data[i] = 0.01f * i;

    // std::cout << "tt_samples:";
    for (int i = 0; i < n_sta * n_grid; ++i)
    {
        tt_samples[i] = 1;
        // std::cout << tt_samples[i] << " ";
    }
    // std::cout << std::endl;

    for (int i = 0; i < n_fm * n_grid * n_sta; ++i)
        intensity[i] = 0.0000001f * i;
    // std::cout << "intensity:" << std::endl;
    // for (int i = 0; i < n_fm * n_grid * n_sta; ++i)
    // {
    //     std::cout << intensity[i] << " ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < n_grid * n_fm * n_samples; ++i)
    //     result[i] = 0.0f;
    // std::cout << "result:";
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << result[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << "Start stackCUDA" << std::endl;
    // Call the stackCUDA function
    stackCUDA(data, tt_samples, intensity, n_sta, n_samples, n_grid, n_fm, result);
    std::cout << "End stackCUDA" << std::endl;
    // Print a portion of the result
    std::cout << "result:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] data;
    delete[] tt_samples;
    delete[] intensity;
    delete[] result;
}

void testIntensity()
{
    std::cout << "CUDA Version: " << CUDART_VERSION << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << ", sizeof(int): " << sizeof(int) << std::endl;

    // Example usage
    int n_sta = 4; // Number of stations
    int n_fm = 1;  // Number of frequency modes

    float *fm_grid = new float[n_fm * 3];
    fm_grid[0] = 339.0f; // Strike
    fm_grid[1] = 15.0f;  // Dip
    fm_grid[2] = 161.0f; // Rake
    GridConfig conf;
    conf.GridSpacingX = 1.0f;
    conf.GridSpacingZ = 1.0f;
    conf.SearchSizeX = 3;
    conf.SearchSizeY = 3;
    conf.SearchSizeZ = 3;
    conf.SearchOriginX = -1.0f;
    conf.SearchOriginY = -1.0f;
    conf.SearchOriginZ = -1.0f;
    int n_grid = conf.SearchSizeX * conf.SearchSizeY * conf.SearchSizeZ; // Number of grid points

    float *stations = new float[n_sta * 3];
    for (int i = 0; i < n_sta * 3; ++i)
        stations[i] = i * 0.001 + 1.0f;

    float a = 0.1f;

    float *result = new float[n_fm * n_grid * n_sta];

    std::cout << "Start intensityCUDA" << std::endl;
    intensityCUDA(fm_grid, &conf, stations, n_sta, n_fm, a, result);
    std::cout << "End intensityCUDA" << std::endl;

    // Print a portion of the result
    std::cout << "result:" << std::endl;
    for (int i = 0; i < 10; ++i)
        std::cout << result[i] << " ";
    std::cout << std::endl;

    // Free host memory
    delete[] fm_grid;
    delete[] stations;
    delete[] result;
}

int main()
{
    testStack();
    // testIntensity();
    return 0;
}