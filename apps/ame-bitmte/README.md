# Native BitMTE 裸机 GEMM：当前接口与四种算例

本目录提供四个可独立构建、运行的精度算例：`i8-fp2pack4`、`i4-fp2pack4`、`i8-i1pack8`、`i4-i1pack8`。它们共享打包代码、固定矩阵寄存器的 GEMM 内核和独立标量校验，以 `EXAMPLE` 选择生成的裸机程序。每个算例均可配置 M/N/K，计算：

```text
C[M][N] = A[M][K] × B[K][N]
```

输出是 i32 的原始 32 位位型，`msaten=0`，每项累加按模 2^32 回绕。内核用 `mzero` 初始化 acc，覆盖原 C；不是 `C += A×B`。本目录不提供 alpha/beta、C 语言矩阵值类型、自动寄存器分配或自动 spill ABI。

## 1. ISA 合同：类型、配置与指令

以下是软件可见的架构语义；后面的固定寄存器规模和调度是本算例的实现选择。完整定义以环境中的 [riscv-matrix-spec](../../../riscv-matrix-spec/spec/program_model.adoc)、[乘法](../../../riscv-matrix-spec/spec/matmul.adoc)、[访存](../../../riscv-matrix-spec/spec/load-store.adoc) 和 [同步](../../../riscv-matrix-spec/spec/zmasync_appendix.adoc) 为准。

### 扩展和类型

`xbitmte` 是依赖 `zames` 的实验扩展，包含全部四种组合；异步使用另需 `zmasync`。新 packed 表示替代旧 fp2pack4 在 load 时展开为 i8 的约定，二者不兼容；没有切换旧表示的 CSR。启用编译选项只说明工具链允许生成这些指令，运行平台也必须实现相应扩展。

`mcfg0..3` 分别配置 `tr0..3`，`mcfg4..7` 分别配置 `acc0..3`。`msetcfg mcfgN, GPR` 写入原始配置值；低 4 位是 `type_code`，位 `[7:4]` 是 `table_sel`。本算例使用 table 0：

| 类型 | type_code | 数值与物理存储 |
| --- | ---: | --- |
| signed i4 | 0 | -8..7，4bit 补码，每字节低 nibble 对应较小 K |
| signed i8 | 2 | -128..127，8bit 补码 |
| signed i32 | 4 | 累加结果 |
| fp2pack4 | 13 | 物理 2bit 三值：`00=0, 01=+1, 11=-1`；`10` 的数值结果为 UB |
| i1pack8 | 15 | 物理 1bit：`0=0, 1=+1`，不是 -1/+1 |

fp2pack4 的名字不表示浮点运算。`packedMode` 只是 CPU/CUTE/NEMU event 的内部元数据，不是 ISA 字段、CSR 或程序需要设置的模式。

### `mmacc` 与 K 分片

```asm
mmacc acc0, tr1, tr0, 2   # 目标 C、来源 B、来源 A、kslice 立即数
mmacc acc0, tr1, tr0      # 等同于显式 selector 0
```

第四操作数 `kslice` 位于指令 `[25:23]`，是编译时常量 0..7。类型组合还会限制可执行的范围；不能用普通 GPR 直接提供运行时 selector。`hw.c` 用 switch 选择带字面量立即数的指令。三个操作数的形式保留旧的零 selector 编码。

令 A 的位宽为 `EA`、B 的 packed 位宽为 `EB`、TR 每行位宽为 `TRLEN`：

```text
Kslot     = TRLEN / EA       # 每条 mmacc 的 A 长度
Kresident = TRLEN / EB       # 一行 packed B 实际容纳的 K 长度
Nslices   = EA / EB

C[i,j] += sum(x=0..Kslot-1) A[i,x] * B[j,kslice*Kslot+x]
```

A 总是从当前 TR 的 K=0 开始。计算下一片时，软件换入相应的 A 分片，B 保持 packed 驻留。`mtilek` 配置本次完整 `Kslot`，不控制 packed B load 的物理长度；本首版合同不支持缩短 K 窗口或部分 M/N 的 native packed 计算。

native packed 的非法控制包括：缺少扩展、类型组合不支持、寄存器类别错误、A/B 同 TR、非完整 shape、越界 selector；其他类型组合的非零 selector 也非法。packed B 仅支持普通 `mlb`，不支持其 typed 转置 load、其他角色的 typed load 或 typed store；raw whole 的语义独立于类型。这些新增规则不改变普通 MTE 的合法性、矩阵权限、地址翻译或访存异常。三值 `10` 是数值 UB，不要求硬件检测或陷入。

### 普通访存与 whole

```asm
mla tr0,  (a_base), a_stride   # 从每个 A 行取一个完整物理行
mlb tr1,  (b_base), b_stride   # 原样取 packed B 行，不解压
mlc acc0, (c_base), c_stride   # 若需要以已有 C 初始化累加器
mzero acc0                    # 本算例采用清零初始化
msc acc0, (c_base), c_stride   # 结果写回
```

stride 以字节为单位。packed B 的内存是 `[N][K]`，行内 K 连续；普通 `mlb` 不做转置。数学 B 若原本是 `[K][N]`，软件须预先准备 `bt[j][k]=B[k][j]` 的布局。

packed `mlb` 每行始终拷贝 `TRLEN/8` 字节。首版完整 A 行也拷贝 `TRLEN/8` 原始字节，i4 保留 nibble 打包。whole 是完整寄存器的原始图像，`rs2=x0`，汇编不写 stride；本内核使用普通带 stride 指令，便于从较大矩阵中取 panel。当前平台可用的 raw whole 子集是 `mla.whole`、`mlb.whole`、`mlc.whole`、`msc.whole`，本算例不依赖 whole。

### 可见性与完成

```asm
msyncregreset sync0       # 开始本次独占任务，token 清零
mfence                   # CPU 已准备的输入对后续矩阵访存可见
# 若干 mla/mlb/mmacc/msc，可异步排队，不逐条等待
mrelease sync0           # 先前矩阵 store 可见后 token 增加
li t0, 1
macquire sync0, t0        # 等待 token >= 1，之后 CPU 才读 C
```

`mfence` 不等同于矩阵计算完成。内核只在全部输出 store 发出后等待一次，函数返回后标量校验读取 C。这里复用 sync0 的前提是上一调用已经完成，且没有其他代码使用它。

`msaten=1` 的溢出结果在此实验版本中由实现定义，NEMU 和 RTL 可能不同；算例明确设为 0。`minit` 只建立矩阵状态，不负责清寄存器、配置或 token，因此内核显式配置类型/shape、清 acc 和重置 sync0。

## 2. 本算例的固定规模与内存布局

本实现面向每行 512bit、128 行的 TR，以及能保存 128×128 i32 的 acc；这组数值不是 ISA 对任意实现的普遍要求。使用 `tr0..3`、`acc0..3`，每个 TR 为 8KiB，每个 acc 为 64KiB。

| EXAMPLE | A | B | 单条计算的 M×N×Kslot | 每 B 行 Kresident | 合法 kslice | 可配置 K 的基数 |
| --- | --- | --- | --- | ---: | --- | ---: |
| `i8-fp2pack4` | signed i8 | 三值 | 128×128×64 | 256 | 0..3 | 256 |
| `i4-fp2pack4` | signed i4 | 三值 | 128×128×128 | 256 | 0..1 | 256 |
| `i8-i1pack8` | signed i8 | 0/1 | 128×128×64 | 512 | 0..7 | 512 |
| `i4-i1pack8` | signed i4 | 0/1 | 128×128×128 | 512 | 0..3 | 512 |

M、N 必须为正的 128 倍数，K 必须为表内基数的正倍数。这是本软件内核的输入约束：它每次消耗完整 B 行，不为尾 K 补零；不是新增的硬件地址异常规则。M/N 的 tile 数可以是奇数，不要求 256 的倍数。需要其他维度时，调用者可先补零、最后裁剪，本目录不会偷偷把非法尾 tile 交给硬件。

输入和输出的连续内存布局：

| 缓冲 | 大小 | 行跨度（字节） |
| --- | --- | --- |
| packed A | `M*K*EA/8` | `a_stride=K*EA/8` |
| packed B，即 bt | `N*K*EB/8` | `b_stride=K*EB/8` |
| C | `M*N*4` | `c_stride=N*4` |

三个缓冲相互不重叠，基址 64B 对齐，全部行都完整可访问。上述 shape 保证 stride 为正的 64B 整数倍。裸机示例使用静态、物理连续内存，不增加虚拟散页翻译或修改已有 MMU/DMA/异常行为。所选规模还需装得进平台 RAM。

低位优先的字节例子：

```text
signed i4 [-8, 7]       -> 0x78
ternary   [0,+1,-1,0]   -> 0x34
binary    [1,0,1,0,0,0,0,1] -> 0x85
```

`bitmte_pack_a/b` 把一个数值行打包；`bitmte_unpack_a/b` 可用于调用方的数据检查。它们返回 0 或 -1；B 解包函数拒绝 UB 编码只是软件 helper 的行为，不是硬件异常承诺。GEMM 不逐项扫描 packed 数据。

## 3. 寄存器分配和指令编排

内核按最多 256×256 的输出宏块遍历，每个宏块的四个 128×128 输出块一直留在 acc 中，跨所有 K 块累加，最后才写回：

| 寄存器 | 用途 |
| --- | --- |
| tr0 | 第 0 组 128 个 A 行的当前 K 分片 |
| tr2 | 第 1 组 128 个 A 行的当前 K 分片 |
| tr1 | 第 0 组 128 个 B 列的完整 packed Kresident |
| tr3 | 第 1 组 128 个 B 列的完整 packed Kresident |
| acc0 / acc1 | 第 0 组 A 行 × 第 0 / 1 组 B 列 |
| acc2 / acc3 | 第 1 组 A 行 × 第 0 / 1 组 B 列 |

每个 Kresident 段内只加载一次 tr1/tr3。每片 A 在两个 N tile 间复用，每个 B 在两个 M tile 间复用；B 还跨所有 slice 复用。完整 2×2 宏块、每 Kresident 段发出 2 次 `mlb`、`2*Nslices` 次 `mla` 和 `4*Nslices` 次 `mmacc`。相较四个输出 tile 各自计算，A/B panel load 数各减半；C 不随 K 分块反复搬进搬出。这只是指令/字节数量分析，不是实测加速比。

例如 i8×fp2pack4 在 `kslice=0` 的编排为：

```asm
# tr1/tr3 已加载 B，tr0/tr2 已加载 A 的第 0 片；stride 是完整矩阵行跨度
mmacc acc0, tr1, tr0, 0
mmacc acc1, tr3, tr0, 0
mla tr0, (a0_next_slice), a_stride  # 排队预取 A0 的第 1 片
mmacc acc2, tr1, tr2, 0
mmacc acc3, tr3, tr2, 0
mla tr2, (a1_next_slice), a_stride  # 排队预取 A1 的第 1 片
# 下一轮执行相应 selector 1；最后一片不再预取
```

同一个 TR 的覆盖排在其全部消费者之后，依赖现有任务调度器处理 RAW/WAR/WAW。把下一片 A load 提前排到另一组 A 的 compute 前，给访存与计算重叠留下机会；不在每条 mmacc 后插入 acquire，也不在 CPU 上反复改写 DMA 仍可能读取的 scratch buffer。宏块边界对 acc 的清零/复用同样依赖先前 store 的读依赖。

四种原生低比特组合都使用 BMTE，单个 BMTE 的计算任务串行；四个 acc 不代表四个并行引擎。通用 MTE 可以独立运行，但这四个纯低比特 GEMM 不人为展开权重来占用它。是否实际隐藏加载延迟、窗口/共享 DMA 的瓶颈及吞吐，需要后续 RTL 测量；NEMU 的顺序功能模拟不能证明这些性能性质。

## 4. 构建与运行四个算例

在 xsai-env 根目录使用原有环境和运行入口，无需 fixture manifest、Python 生成器、外部 golden 文件或新的根包装：

```sh
source env.sh

# 算例 1：i8 × 三值，M/N/K 可分别改为 128/128/256 的正倍数
make -C nexus-am/apps/ame-bitmte -j4 image EXAMPLE=i8-fp2pack4 M=256 N=256 K=512
make run-nemu PAYLOAD="$PWD/nexus-am/apps/ame-bitmte/build/i8-fp2pack4/256x256x512/gemm.bin"

# 算例 2：i4 × 三值
make -C nexus-am/apps/ame-bitmte -j4 image EXAMPLE=i4-fp2pack4 M=384 N=128 K=768
make run-nemu PAYLOAD="$PWD/nexus-am/apps/ame-bitmte/build/i4-fp2pack4/384x128x768/gemm.bin"

# 算例 3：i8 × 0/1，K 改为 512 的正倍数
make -C nexus-am/apps/ame-bitmte -j4 image EXAMPLE=i8-i1pack8 M=128 N=384 K=1024
make run-nemu PAYLOAD="$PWD/nexus-am/apps/ame-bitmte/build/i8-i1pack8/128x384x1024/gemm.bin"

# 算例 4：i4 × 0/1
make -C nexus-am/apps/ame-bitmte -j4 image EXAMPLE=i4-i1pack8 M=384 N=384 K=512
make run-nemu PAYLOAD="$PWD/nexus-am/apps/ame-bitmte/build/i4-i1pack8/384x384x512/gemm.bin"
```

省略 M/N 时均为 256；省略 K 时三值为 512、二值为 1024。每个模式、每组 shape 各有独立对象目录和 `gemm.{elf,bin,txt}`，切换参数不需要 clean。非法 M/N/K 在编译期诊断；API 还在发指令前检查软件参数。`make -C nexus-am/apps/ame-bitmte help` 显示用法。

使用环境提供的自定义 LLVM，app 通过现有 `Makefile.app` 和 `riscv64-xs` 路径构建。`-march` 包含 `zames_zmasync_xbitmte`；最终反汇编 `gemm.txt` 包含 selector 指令。NEMU 需为现有 AME/Xbitmte 开启的 standalone 版本，并从 0x80000000 运行本裸机程序。

程序生成确定性 signed A 和合法 B，先打包，再运行矩阵内核，最后用**原始未打包数值**逐个校验全部 M×N 输出；标量参考不调用 unpack 或复用内核的分块/selector 逻辑。A 包含 i4/i8 的正负极值，三值 B 包含 -1/0/+1。C 预填非零哨兵以检查覆盖。成功打印例如：

```text
PASS i8-fp2pack4 M=128 N=128 K=256 elements=16384 checksum=fffdd5ae
```

随后 NEMU 应报告 `HIT GOOD TRAP`、退出码 0；任何结果不一致都打印首个坐标并以非零状态退出。校验和方便辨认运行对象，正确性判断仍逐元素进行。

## 5. 在其他裸机程序中复用

```c
#include "bitmte.h"

/* 由调用者分配对齐的 packed_a[M][K*EA/8]、packed_bt[N][K*EB/8]、c[M][N]。
 * 每行通过 bitmte_pack_a/b 准备，或直接提供相同位序的合法 packed 数据。
 */
bitmte_init();
int status = bitmte_gemm(&c[0][0], &packed_a[0][0], &packed_bt[0][0],
                         M, N, K, BMTE_I8_TERNARY);
/* status==0 后 C 已完成并可被 CPU 读取。 */
```

`bitmte_init` 面向 M 态裸机，启用 MS 并执行 minit。内核独占全部 TR/acc、对应 mcfg、tile CSR、msaten 和 sync0，不保存恢复调用者的矩阵状态，也不支持重入或其他矩阵使用者同时干扰；普通 C 参数只传指针和 GPR 元数据。inline asm 使用 volatile 和 memory clobber，编译器不自动分配/保存矩阵寄存器。

C 使用 `uint32_t` 保存 i32 的精确位型；CPU 参考以 unsigned 加法得到定义良好的模 2^32 结果，避免 signed C 溢出的 UB。pack/unpack 和数值准备只需进行一次；实用推理程序可直接复用预打包权重。示例为检查正确性还保留未打包 A/B 并执行标量 GEMM，因此整个程序的宿主耗时不能当作加速器性能。

## 6. 目录范围

- `main.c`：四种构建变体共用的确定性数据、独立全量标量校验及裸机入口。
- `bitmte.h` / `bitmte.c`：数值类型、几何和行级 pack/unpack 接口。
- `hw.c`：固定寄存器 inline asm、2×2 阻塞 GEMM、同步。
- `Makefile`：现有 AM 构建路径、模式/尺寸选择、独立产物目录。

本目录交付只包含这些代码和本文档。历史 fixture/非法指令/host/perf/RTL 开发检查脚本已外部归档，不参与正常构建。验证记录在环境的 `docs/bitmte-execution/`，临时验证程序和原始日志仅留在忽略的 build/log 目录；不把历史测试结果当作本版新内核的测试结果。
