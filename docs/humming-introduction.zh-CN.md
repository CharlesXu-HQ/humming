# Humming：高性能、灵活的量化推理内核

Humming 由蚂蚁集团 Venus 团队与 vLLM 社区合作开发，是面向 Dense 与混合专家（MoE）量化推理的高性能 JIT 编译 GEMM 内核库。通过统一的量化表示、可复用的 CUDA 组件与面向不同设备的深度调优，Humming 将丰富的量化格式与广泛的硬件支持整合到同一套框架中，让更多模型与设备受益于高效、灵活的量化推理。

## 背景

随着模型规模与推理负载的增长，显存容量、访存带宽和计算成本日益成为部署的制约因素。这一问题在 MoE 模型中尤为明显：每个 token 只激活部分专家，但仍需存储全部权重。例如，DeepSeek-V3 拥有 6710 亿参数，每个 token 激活其中的 370 亿。量化通过降低权重与激活的表示精度，减少显存占用和数据搬运，也有助于提高计算效率。

不同模型和 GPU 架构的量化需求各不相同。常见配置包括 WNA16、W8A8、W4A8 和 W4A4，其中 W 和 A 分别标识权重与激活的位宽，N 表示可变的权重位宽。即使位宽相同，INT4 和 NVFP4 等方案在数值格式和缩放方式上也存在差异。缩放粒度、缩放因子的表示方式、分组大小，以及对称或非对称量化进一步扩大了配置空间。

这些存储与计算上的收益需要通过高效内核，才能转化为更低的延迟和更高的吞吐。Prefill、Decode 和不同的 MoE 专家负载对计算与访存有不同要求。内核既要适应这些负载，也要控制数据转换、反量化和调度的开销。

为每种组合开发并接入专用内核，会带来不断增长的维护负担。每新增一种组合，都可能需要独立的权重预处理、内核选择、框架适配和测试，已有优化也需要同步到多个实现中。

Humming 将量化描述与执行策略分离，并通过共享组件构建内核，以应对这一问题。

## 通用架构

Humming 的执行路径包含三个环节：表示量化计算，根据硬件与负载选择配置，再通过 CUDA 组件编译生成内核。

### 统一量化表示

不同量化工具导出的模型在数值格式、参数组织和权重布局上各有差异。Humming 将支持的导出格式转换为 Humming 标准格式：一组通用的量化描述，以及按统一约定组织的权重、缩放因子、零点和其他张量。

<p align="center">
  <a href="humming-quantization-format-en.svg"><img src="humming-quantization-format-en.svg" alt="Humming 统一量化表示与参数组合" width="700"></a>
</p>

这一表示保留各方案的量化语义，将权重格式、缩放粒度、缩放因子的数据类型、零点和二级缩放因子分别表示为独立参数。激活拥有自己的数值格式与量化设置。这些参数可以在格式与硬件约束内组合。

例如，采用分组缩放的 INT4 权重可以搭配 BF16 激活；采用不同缩放方式的低精度浮点权重也使用同一套参数体系。这为推理框架处理不同来源的模型提供了统一接口。

### 按负载调优与 JIT 编译

标准格式为不同导出格式提供统一表示。随后，预处理完成参数校验、维度对齐，并将这些张量重排为适合目标内核的布局。Humming 结合量化描述、权重形状、GEMM 类型、目标 GPU 和计算选项，确定执行配置。

<p align="center">
  <a href="humming-jit-inference-en.svg"><img src="humming-jit-inference-en.svg" alt="Humming 预处理、调优、JIT 编译与推理流程" width="700"></a>
</p>

内核由三类配置描述：

- **Layer Config** 描述张量形状、数据类型、量化参数和目标架构。
- **Compute Config** 指定 GEMM 类型和计算选项，例如批次不变性与输入缩放布局。
- **Tuning Config** 选择分块形状、流水线级数、Warp 分工模式、访存方式和 MMA 指令，同时控制并发与调度；TMA 可按张量独立配置。

默认情况下，Humming 根据设备特定的启发式规则选择调优配置。它可以为不同输入规模区间准备不同配置，并在运行时分派到合适的内核。

Humming 的统一架构使其能够方便地为不同负载调整执行参数、选择流水线。在小批量下，使用异步拷贝（`cp.async`）和 MMA 的多级流水线可能优于基于 TMA 与 UMMA 的流水线；在大批量下，TMA 与 UMMA 则可能提供更高的吞吐。Humming 在同一套配置与组件框架内支持这两种执行路径，从而为不同负载选择更有效的策略。

这三类配置共同用于特化由 NVRTC 或 NVCC 编译的内核，编译结果会被缓存以供复用。推理时，选定的内核使用预处理后的权重和运行时激活完成计算。

### 跨格式与架构共享 CUDA 组件

在 CUDA 层，Humming 将任务调度、数据搬运、反量化与缩放、Tensor Core 计算和输出处理组织为可复用组件，并提供常规流水线、Warp Specialization 流水线和 UMMA 专用流水线。这些流水线共享适用的组件，同时采用各自执行策略所需的数据流与同步方式。

<p align="center">
  <a href="humming-cuda-architecture-en.svg"><img src="humming-cuda-architecture-en.svg" alt="Humming CUDA 共享结构与可替换实现" width="700"></a>
</p>

各组组件分别处理不同来源的差异：

- **数值处理组件适配量化格式。** 解码与算术组件处理位宽、整数或浮点表示、缩放因子和零点等差异，反量化与缩放融入内核计算过程。
- **访存与计算后端适配 GPU 架构。** Humming 根据目标架构选择数据搬运与 Tensor Core 实现，并在支持的场景中使用异步拷贝、TMA 和多播。
- **任务映射与调度适配 Dense 和 MoE 负载。** 每种 GEMM 类型采用相应的调度逻辑，缩放和偏置加法也可以融合到输出阶段。

JIT 编译依据三类配置组合这些组件，并消除未使用的实现分支。因此，支持新格式或新架构可以通过扩展相关组件完成。共享组件的改进能够使适用的配置受益，减少多套专用内核带来的重复开发与维护。

### 基于设备参数的性能调优

Humming 结合 GEMM 配置与设备参数，对内核进行深度优化。从量化格式、矩阵形状到 GPU 的指令能力、SM 数量、共享内存与寄存器资源，Humming 将这些因素共同纳入执行策略，让低比特存储的优势转化为实际的推理性能。

**让已有 GPU 也能高效运行 W4A8-MXFP4。** Blackwell 之前的架构缺少原生 MXFP4 Tensor Core 计算支持，常规 W4A8-MXFP4 内核在 M 较大时容易受到反量化与缩放开销的限制。为此，Humming 设计了 MXFP4 到 INT8/FP8 的快速反量化路径，在内核中将 MXFP4 权重高效转换为设备支持的计算格式。在这一场景下，该路径在小 M 时明显优于常规 W4A8 实现，在大 M 时则与 W8A8 的性能基本持平。这使 H200 等 GPU 也能在保留 4-bit 权重存储优势的同时，运行高性能的 W4A8-MXFP4 算子。

**为不同设备选择合适的反量化路径。** 相同的量化格式，在不同 GPU 上也可能需要不同的实现。Humming 根据设备能力选择解码、数值转换与缩放路径，并与对应的访存方式和计算流水线配合，减少反量化过程中的指令与数据搬运开销。共享组件使这些优化能够复用于适用的 Dense 与 MoE 内核。

**用丰富参数与设备专属启发式适配不同负载。** Humming 开放分块形状、流水线级数、Warp 分工、TMA、MMA 指令、CTA 并发度及 Stream-K 调度等调优参数。针对不同设备，启发式逻辑结合 GEMM 形状与资源约束，在并行度、数据复用和流水线开销之间选择合适的配置，并随输入规模切换执行策略。开发者既可以直接使用默认配置，也可以针对特定模型和部署场景进一步调优。

## 使用方法

### 安装

通过 pip 安装 Humming：

```bash
pip install humming-kernels
```

Humming 使用 CUDA 编译器按需生成内核。如果系统中没有可用的 CUDA 编译器，可以根据环境的 CUDA 主版本，选择以下一种方式一并安装编译依赖：

```bash
pip install "humming-kernels[cu12]"  # CUDA 12
pip install "humming-kernels[cu13]"  # CUDA 13
```

### 在 vLLM 中使用

Humming 已与 vLLM 深度集成。用户可以通过 `--linear-backend` 和 `--moe-backend`，分别选择 Humming 作为 Dense GEMM 和 MoE GEMM 的计算后端。例如，同时为两类计算启用 Humming：

```bash
vllm serve openai/gpt-oss-20b --linear-backend humming --moe-backend humming
```

对于 vLLM 尚未原生支持解析、但 Humming 支持的量化 checkpoint，也可以尝试通过 `--quantization humming` 显式选择 Humming 的量化加载路径：

```bash
vllm serve openai/gpt-oss-20b --quantization humming
```

### 选择激活类型

默认情况下，Humming 沿用 checkpoint 中设定的激活类型，并根据设备能力自动适配。如果 GPU 不支持对应的计算类型，Humming 会回退到设备支持的更高精度类型。例如，在不支持原生 MXFP4 计算的设备上，可以自动选择 FP8、BF16 或 FP16 计算路径，让同一份量化 checkpoint 适配不同代际的 GPU。

用户也可以通过 vLLM 的量化配置接口覆盖默认设置，按部署需求选择激活类型。以下示例将 MoE 层的激活类型指定为 MXFP8：

```bash
vllm serve openai/gpt-oss-20b --quantization-config.moe.activation mxfp8
```

## 结果

### 广泛的设备支持

得益于统一架构，Humming 能够覆盖广泛的 GPU 设备。对于 NVIDIA GPU，Humming 完整支持从 Turing 起的所有 GPU 架构，让不同代际的设备都能使用同一套量化计算框架。

统一的 JIT 架构也让 Humming 能够方便地扩展到兼容 CUDA 编程的其他 GPU。目前，Humming 已完整支持平头哥 Zhenwu 810E，将统一的内核与调优能力带到更多硬件平台。

下图列出了 Humming 在各类设备上支持的高性能激活类型，即能够充分发挥对应设备最高计算性能的类型。部分未列出的类型仍可通过其他指令路径运行。例如，B300（SM103）的 UMMA 指令不再支持 INT8 计算，但仍可使用较早的 MMA 指令执行 INT8 GEMM，只是性能会明显降低。

<p align="center">
  <a href="humming-device-support-en.svg"><img src="humming-device-support-en.svg" alt="Humming 设备架构与激活类型支持矩阵" width="700"></a>
</p>

### 量化支持

Humming 对量化类型的支持主要取决于激活类型与权重类型是否兼容。统一的量化表示与内核架构，让多种整数和浮点权重格式能够搭配不同精度的激活参与计算，为模型部署提供灵活的精度与性能选择。

下图展示了 Humming 支持的权重类型与激活类型组合：行表示权重类型，列表示激活类型，勾选表示支持对应组合。结合上一节的设备支持矩阵，即可了解各类硬件上可用的高性能量化计算路径。

<p align="center">
  <a href="humming-quantization-support-en.svg"><img src="humming-quantization-support-en.svg" alt="Humming 权重与激活类型支持矩阵" width="700"></a>
</p>

Humming 还支持一种指数位数为 0 的特殊浮点格式，其取值范围等价于相同位宽的对称整数。例如，排除 `-128` 的 INT8 与 `float8e0m7` 都表示 `-127` 到 `127` 的整数。这类权重与激活类型的兼容性等同于相同位宽的 INT 权重。对于符合转换条件的对称整数权重，Humming 会自动将其转换为这种格式，以加快反量化。

### 性能

下图比较各 GPU 上 Dense W4A16 GEMM 相对 cuBLAS 的加速比，M 覆盖 1 到 8192 的所有 2 的幂。

<p align="center">
  <a href="humming-w4a16-performance-en.svg"><img src="humming-w4a16-performance-en.svg" alt="各 GPU 上 W4A16 GEMM 相对 cuBLAS 的加速比" width="1200"></a>
</p>

下图展示 MoE W4A16 内核相对 vLLM Marlin 的加速比，采用 256 个专家、top_k=8 和 BF16 激活，M 同样覆盖 1 到 8192 的所有 2 的幂。Gateup 与 Down 分别在冷 L2 缓存下测量，再将两项耗时相加计算加速比；图例标明各内核采用的 Indexed 或 Grouped 布局。

<p align="center">
  <a href="humming-w4a16-moe-performance-en.svg"><img src="humming-w4a16-moe-performance-en.svg" alt="MoE W4A16 内核的 Gateup 与 Down 合计耗时相对 vLLM Marlin 的加速比" width="1200"></a>
</p>

TODO: W4A8 / W8A8 / W4A4

## 未来计划

### 更完善的设备支持

Humming 将继续扩展统一架构的硬件覆盖，并深入适配新设备的计算与访存能力，让更多平台充分释放量化推理的性能潜力。接下来的重点包括：

- 支持 Rubin 架构的更多新特性，包括 Locality Domains，进一步优化任务映射与数据访问。
- 扩展对更多平头哥 PPU 设备的支持，包括 ZW890P。
- 支持更多兼容 CUDA 编程的设备，将已有的量化格式、内核组件与调优能力带到更多硬件平台。

### 添加 Autotune

经过针对性调优，Humming 已能在多数场景达到 SOTA 级别的性能。但模型形状、量化配置与设备资源的组合十分丰富，启发式逻辑难以在所有情况下都选出最优配置。

为此，Humming 计划引入 autotune，通过在目标设备上实测候选配置，为具体负载寻找更合适的执行参数。自动调优将与现有启发式互补，减少手动搜索配置的成本，让更多场景接近内核的最佳性能。
