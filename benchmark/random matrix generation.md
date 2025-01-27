# 随机矩阵生成基准测试报告

## 基准测试环境

### 环境A

- **硬件**:
  - CPU: AMD Ryzen 8 8845H @4.6GHz (8 cores)
  - GPU: NVIDIA GeForce RTX 4070 laptop GPU @2625MHz with 8GB GDDR6 @18Gbps
  - 内存: 16GB $\times$ 2 DDR5 @5600MHz
  - 存储: 1TB NVMe SSD

- **软件**:
  - 操作系统: Windows 11 Pro for Workstations 64-bit
  - 编译器: Microsoft Visual Studio 2022
  - CUDA Toolkit: 12.8.61
  - cuBLAS: 12.8.3.14
  - cuRAND: 10.3.9.55

### 环境B

- **硬件**:
  - CPU: AMD EPYC 9654 @3.7GHz (96 cores)
  - GPU: NVIDIA GeForce RTX 4090D @2520MHz with 24GB GDDR6X @21Gbps
  - 内存: 32GB $\times$ 6 DDR5 @4800MHz
  - 存储: 3.84TB NVMe SSD

- **软件**:
  - 操作系统: Windows Server 2025 Datacenter 64-bit
  - 编译器: Microsoft Visual Studio 2022
  - CUDA Toolkit: 12.8.61
  - cuBLAS: 12.8.3.14
  - cuRAND: 10.3.9.55

## 基准测试方法

- **输入参数**:
  - 矩阵大小: 5000 $\times$ 5000
  - 矩阵类型: 随机矩阵
  - 数据类型: int, float, double
  - 重复次数: 1000

  ```cpp
  CudaMatrix<typename T> A(5000, Random);
  ```

- **测试步骤**:
  1. 初始化CUDA环境。
  2. 创建1000次5000 $\times$ 5000的随机矩阵。
  3. 记录创建矩阵所需的时间，取1000次平均值。

- **测量指标**:
  - 矩阵创建时间（秒）

## 基准测试结果

**表1：** 不同环境下的随机矩阵创建性能

| **矩阵类型** | **环境A用时（秒）** | **环境B用时（秒）** |
| :---: | :---: | :---: |
| int | 5.401 | 1.703 |
| float | 5.390 | 1.698 |
| double | 5.392 | 1.700 |

## 结果分析

- **创建时间**:
  - 环境A创建5000 $\times$ 5000的随机矩阵耗时约5.39秒，环境B耗时约1.7秒。
  - 在环境A下，较在核函数内执行类型判断和随机数生成，矩阵创建的性能提高3倍。
  - 性能差异主要与核心频率、内核优化程度、随机数生成器效率相关。  
  - 建议使用更高效的随机数生成器、批量生成与流并行化技术，并优化内核函数以减少线程同步和内存访问。  
  - int、float、double性能相近，int型略慢或与初始内核频率较低所致。

- **优化方向**:
  - 使用更高效的随机数生成器。
  - 优化内核函数，减少线程间的同步和内存访问开销。
  - 使用批量生成和流并行化技术。

## 结论

本次基准测试展示了在当前硬件和软件配置下，创建5000 x 5000随机矩阵的性能。4070 laptop创建5000 $\times$ 5000的随机矩阵耗时约5.39秒，4090d耗时约1.7秒。通过进一步优化内核函数和使用更高效的随机数生成器，可以进一步提高性能。未来的工作将集中在优化内核函数和使用批量生成和流并行化技术上。
