# Announcing Humming: Fast, Flexible Kernels for Quantized Inference

Jointly developed by Ant Group's Venus team and the vLLM community, Humming is a high-performance, JIT-compiled GEMM kernel library for quantized dense and mixture-of-experts (MoE) inference. By combining a unified quantization representation, reusable CUDA components, and extensive tuning for different devices, Humming brings diverse quantization formats and broad hardware support into one framework, making efficient, flexible quantized inference available to more models and devices.

## Background

As models and inference workloads grow, memory capacity, memory bandwidth, and compute costs increasingly constrain deployment. This is especially apparent in MoE models: each token activates only a subset of experts, but the full set of weights still needs to be stored. DeepSeek-V3, for example, has 671 billion parameters, with 37 billion activated per token. Quantization lowers the precision of weights and activations to reduce memory usage and data movement, and can also improve compute efficiency.

Quantization requirements vary across models and GPU generations. Common configurations include WNA16, W8A8, W4A8, and W4A4, where W and A specify weight and activation bit widths, and N denotes a variable weight bit width. Even at the same precision, schemes such as INT4 and NVFP4 differ in numeric format and scaling. Scaling granularity, scale representation, group size, and symmetric or asymmetric quantization further expand the configuration space.

These storage and compute savings only translate into lower latency and higher throughput with efficient kernels. Prefill, decode, and varying MoE expert loads place different demands on compute and memory. Kernels must adapt to these workloads while keeping conversion, dequantization, and scheduling overhead low.

Developing and integrating a dedicated kernel for every combination creates a growing maintenance burden. Each addition can require its own weight preprocessing, kernel selection, framework adapters, and tests. Optimizations must then be carried across multiple implementations.

Humming addresses this problem by separating quantization descriptions from execution choices and composing kernels from shared components.

## General-Purpose Architecture

Humming's execution path has three parts: represent the quantized computation, select configurations for the hardware and workload, and compile a kernel from CUDA components.

### A Unified Quantization Representation

Models exported by different quantization tools vary in numeric format, parameter organization, and weight layout. Humming converts supported export formats into the Humming standard format: a common quantization description with consistently organized weights, scales, zero points, and other tensors.

<p align="center">
  <a href="humming-quantization-format-en.svg"><img src="humming-quantization-format-en.svg" alt="Humming unified quantization representation and parameter combinations" width="700"></a>
</p>

This representation preserves each scheme's quantization semantics while describing weight format, scaling granularity, scale data type, zero points, and secondary scales as separate parameters. Activations have their own numeric format and quantization settings. These parameters can be combined subject to format and hardware constraints.

For example, groupwise INT4 weights can be paired with BF16 activations. Low-precision floating-point weights with different scaling schemes use the same parameter system. This gives inference frameworks a common interface for models from different sources.

### Tuning and JIT Compilation for Each Workload

The standard format provides a common representation across export formats. Preprocessing then validates parameters, pads dimensions, and repacks these tensors into layouts suited to the target kernel. Humming combines the quantization description with the weight shape, GEMM type, target GPU, and compute options to configure execution.

<p align="center">
  <a href="humming-jit-inference-en.svg"><img src="humming-jit-inference-en.svg" alt="Humming preprocessing, tuning, JIT compilation, and inference flow" width="700"></a>
</p>

Three configurations describe the kernel:

- **Layer Config** describes tensor shapes, data types, quantization parameters, and the target architecture.
- **Compute Config** specifies the GEMM type and computation options, such as batch invariance and input scale layout.
- **Tuning Config** selects tile shapes, pipeline stage counts, warp specialization, memory access, and MMA instructions. It also controls concurrency and scheduling; TMA can be configured per tensor.

By default, Humming selects tuning configurations using device-specific heuristics. It can prepare different configurations for ranges of input sizes and dispatch to the appropriate kernel at runtime.

Humming’s unified architecture makes it straightforward to tune execution parameters and select pipelines for different workloads. At small batch sizes, a multistage pipeline using asynchronous copies (`cp.async`) and MMA can outperform a pipeline built around TMA and UMMA. At larger batch sizes, TMA and UMMA can offer better throughput. Humming accommodates both execution paths within the same configuration and component framework, allowing it to select the more effective strategy for each workload.

Together, the three configurations specialize kernels compiled with NVRTC or NVCC. Compiled kernels are cached for reuse. Inference combines the selected kernel with the preprocessed weights and runtime activations.

### Shared CUDA Components Across Formats and Architectures

At the CUDA level, Humming organizes task scheduling, data movement, dequantization and scaling, Tensor Core computation, and output processing into reusable components. It provides standard, warp-specialized, and dedicated UMMA pipelines. These pipelines share compatible components while using the dataflow and synchronization required by each execution strategy.

<p align="center">
  <a href="humming-cuda-architecture-en.svg"><img src="humming-cuda-architecture-en.svg" alt="Humming shared CUDA structure and pluggable implementations" width="700"></a>
</p>

Each group of components handles a different source of variation:

- **Numeric processing handles quantization formats.** Decoding and arithmetic components account for bit width, integer or floating-point representation, scales, and zero points. Dequantization and scaling are integrated into the kernel's computation.
- **Memory and compute backends handle GPU architectures.** Humming selects data movement and Tensor Core implementations for the target architecture, using asynchronous copies, TMA, and multicast where supported.
- **Task mapping and scheduling handle dense and MoE workloads.** Each GEMM type uses the appropriate scheduling logic. Scaling and bias addition can also be fused into the output stage.

JIT compilation combines these components according to the three configurations and eliminates unused implementation branches. A new format or architecture can therefore be supported by extending the relevant components. Improvements to a shared component benefit compatible configurations, reducing the duplicated development and maintenance of separate specialized kernels.

### Performance Tuning Based on Device Parameters

Humming deeply optimizes kernels using both the GEMM configuration and device parameters. Quantization formats, matrix shapes, GPU instruction support, SM count, shared memory, and register resources all inform execution choices, turning the benefits of low-bit storage into practical inference performance.

**Efficient W4A8-MXFP4 inference on existing GPUs.** Architectures before Blackwell lack native MXFP4 Tensor Core support, and conventional W4A8-MXFP4 kernels can be limited by dequantization and scaling overhead at larger M. Humming addresses this with a fast MXFP4-to-INT8/FP8 dequantization path that converts MXFP4 weights into a supported compute format within the kernel. In this setting, the path substantially outperforms conventional W4A8 implementations at small M and approaches W8A8 performance at large M. GPUs such as H200 can therefore run high-performance W4A8-MXFP4 kernels while retaining the storage benefits of 4-bit weights.

**Dequantization paths matched to each device.** The same quantization format can benefit from different implementations across GPUs. Humming selects decoding, numeric conversion, and scaling paths for the device, pairing them with suitable memory access and compute pipelines to reduce instruction and data movement overhead. Shared components make these optimizations reusable across compatible dense and MoE kernels.

**Extensive tuning controls with device-specific heuristics.** Humming exposes tile shapes, pipeline stage counts, warp specialization, TMA, MMA instructions, CTA concurrency, and Stream-K scheduling. Device-specific heuristics combine GEMM shapes with resource constraints to balance parallelism, data reuse, and pipeline overhead, selecting different execution strategies as input sizes change. Developers can use the default configurations directly or refine them for particular models and deployment scenarios.

## Usage

### Installation

Install Humming with pip:

```bash
pip install humming-kernels
```

Humming uses a CUDA compiler to generate kernels on demand. If no CUDA compiler is available on the system, choose one of the following installation options to include the compilation dependencies for your environment's CUDA major version:

```bash
pip install "humming-kernels[cu12]"  # CUDA 12
pip install "humming-kernels[cu13]"  # CUDA 13
```

### Using Humming in vLLM

Humming is deeply integrated into vLLM. The `--linear-backend` and `--moe-backend` options select Humming as the compute backend for dense GEMM and MoE GEMM, respectively. For example, enable Humming for both:

```bash
vllm serve openai/gpt-oss-20b --linear-backend humming --moe-backend humming
```

For quantized checkpoints that vLLM does not yet parse natively but Humming supports, users can also try `--quantization humming` to explicitly select Humming's quantization loading path:

```bash
vllm serve openai/gpt-oss-20b --quantization humming
```

### Selecting Activation Types

By default, Humming uses the activation type specified in the checkpoint and adapts execution to the device. If the GPU does not support that compute type, Humming falls back to a supported higher-precision type. For example, devices without native MXFP4 support can automatically use FP8, BF16, or FP16 compute paths, allowing the same quantized checkpoint to run across GPU generations.

Users can override the default through vLLM's quantization configuration interface to select an activation type for their deployment. The following example sets the activation type for MoE layers to MXFP8:

```bash
vllm serve openai/gpt-oss-20b --quantization-config.moe.activation mxfp8
```

## Results

### Broad Device Support

Humming's unified architecture enables broad GPU support. For NVIDIA GPUs, Humming fully supports every architecture from Turing onward, bringing a common quantized computing framework to multiple generations of hardware.

The unified JIT architecture also makes it straightforward to extend Humming to other GPUs compatible with CUDA programming. Humming already fully supports T-Head's Zhenwu 810E, bringing its shared kernels and tuning capabilities to more hardware platforms.

The figure below shows the high-performance activation types supported by Humming on each device: types that can take full advantage of the device's peak compute capabilities. Some types omitted from the figure can still run through other instruction paths. For example, UMMA instructions on the B300 (SM103) no longer support INT8 computation, but earlier MMA instructions can still execute INT8 GEMM, with substantially lower performance.

<p align="center">
  <a href="humming-device-support-en.svg"><img src="humming-device-support-en.svg" alt="Humming device architecture and activation type support matrix" width="700"></a>
</p>

### Quantization Support

Humming's support for quantization types depends primarily on compatibility between activation and weight types. Its unified quantization representation and kernel architecture allow a wide range of integer and floating-point weight formats to work with activations at different precisions, providing flexible choices for model precision and deployment performance.

The figure below shows the weight and activation type combinations supported by Humming. Rows represent weight types, columns represent activation types, and checks indicate supported combinations. Together with the device support matrix above, it outlines the high-performance quantized computation paths available on each hardware architecture.

<p align="center">
  <a href="humming-quantization-support-en.svg"><img src="humming-quantization-support-en.svg" alt="Humming weight and activation type support matrix" width="700"></a>
</p>

Humming also supports a special floating-point format with zero exponent bits, whose representable values match those of symmetric integers at the same bit width. For example, INT8 excluding `-128` and `float8e0m7` both represent integers from `-127` to `127`. Weights in this format have the same activation type compatibility as INT weights of the same bit width. Humming automatically converts eligible symmetric integer weights to this format for faster dequantization.

### Performance

The following figure compares Dense W4A16 GEMM speedups over cuBLAS across GPUs, with M covering every power of two from 1 to 8192.

<p align="center">
  <a href="humming-w4a16-performance-en.svg"><img src="humming-w4a16-performance-en.svg" alt="W4A16 GEMM speedups over cuBLAS across GPUs" width="1200"></a>
</p>

The following figure compares MoE W4A16 kernels against vLLM Marlin, using 256 experts, top_k=8, and BF16 activations, with M again covering every power of two from 1 to 8192. Gateup and Down are measured independently with a cold L2 cache, and their latencies are summed to calculate speedup. The legend identifies each kernel's Indexed or Grouped layout.

<p align="center">
  <a href="humming-w4a16-moe-performance-en.svg"><img src="humming-w4a16-moe-performance-en.svg" alt="MoE W4A16 kernel speedups over vLLM Marlin using summed Gateup and Down latencies" width="1200"></a>
</p>

TODO: W4A8 / W8A8 / W4A4

## Future Plans

### Broader and Deeper Device Support

Humming will continue expanding hardware coverage while adapting to the compute and memory capabilities of new devices, helping more platforms realize the performance potential of quantized inference. Priorities include:

- Supporting more Rubin architecture features, including Locality Domains, to further optimize task mapping and data access.
- Extending support to more T-Head PPU devices, including the ZW890P.
- Supporting more devices compatible with CUDA programming, bringing existing quantization formats, kernel components, and tuning capabilities to additional hardware platforms.

### Adding Autotune

With targeted tuning, Humming can already achieve state-of-the-art performance in most scenarios. However, the wide range of model shapes, quantization configurations, and device resources makes it difficult for heuristics to select the optimal configuration in every case.

Humming therefore plans to introduce autotune, benchmarking candidate configurations on the target device to find better execution parameters for each workload. Autotune will complement existing heuristics, reducing manual configuration search and helping more workloads approach the kernels' best achievable performance.
