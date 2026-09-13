# BF16 × BF16 → FP32 GEMM

本算例实现 `C[m,n] = Σ_k A[m,k] × W[k,n]`，A/W 为 BF16，C 为 FP32。采用 BitVLA attention OpenLLC 已测的 `1M × 2N`、A ping-pong、cross-B prefetch 和 CReg pair 延迟写回调度；QKᵀ、AV 共用同一内核。

## 构建与检查

在已配置自定义 LLVM 的 xsai-env 根目录执行 `source env.sh`。NEMU、XSAI 和 XSAI/CUTE 使用支持 `BF16 × BF16 → FP32` 的 `feat-bitnet` 分支，通过 `zames/zmasync` 汇编矩阵指令。独立使用 nexus-am 时设置 `AM_HOME` 为仓库根目录、`LLVM_HOME` 为该 LLVM 安装目录。

```bash
source env.sh
make -C nexus-am/tests/ame-gemm-bf16-fp32 M=128 N=256 K=32
NEMU/build/riscv64-nemu-interpreter -b nexus-am/tests/ame-gemm-bf16-fp32/build/ame-gemm-bf16-fp32-m128-n256-k32-vf0-riscv64-xs.bin
```

| 参数 | 默认值 | 约束 |
| --- | ---: | --- |
| M | 128 | 正的 128 倍数 |
| N | 256 | 正的 128 倍数 |
| K | 32 | 正的 32 倍数 |
| VERIFY_FULL | 0 | 0：逐输出 tile 抽样及边界检查；1：检查全部输出 |

各尺寸和校验模式使用独立构建目录及 payload 名。输入使用精确可表示的 BF16 `{-1,-0.5,+0.5,+1}`，由行列和 K 坐标确定；标量 expected-value 与 FP32 输出比较。全部初始化和检查在计时之外。矩阵静态分配，尺寸须满足运行平台内存容量；用于精确比对的测试数据还需保持标量累加和 FP32 表示在适用范围内。

两个原始 attention 形状的构建命令：

```bash
# QK^T: [1024,128] × [128,1024]
make -C nexus-am/tests/ame-gemm-bf16-fp32 M=1024 N=1024 K=128
# AV: [1024,1024] × [1024,128]
make -C nexus-am/tests/ame-gemm-bf16-fp32 M=1024 N=128 K=1024
# 最小 tile 完整校验
make -C nexus-am/tests/ame-gemm-bf16-fp32 M=128 N=128 K=32 VERIFY_FULL=1
```

## 内存布局

数学右操作数为 `W[K,N]`，加载视图为 `B_storage[N,K]`，且 `B_storage[n,k] = W[k,n]`。内存中 A 和 B_storage 都是行主序，C 也是行主序：

| 对象 | 存储维度 | 元素字节偏移 | load/store stride |
| --- | --- | --- | --- |
| A | `[M,K]` | `2*(m*K+k)` | `2K` B |
| B_storage | `[N,K]` | `2*(n*K+k)` | `2K` B |
| C | `[M,N]` | `4*(m*N+n)` | `4N` B |

三个数组均 64 B 对齐。每条 B tile load 读取 128 个 N row，每行读取连续 32 个 BF16（64 B）；相邻行首相距 `2K` 字节。一个 tile 的命令 payload 为 8 KiB，但 K>32 时该 tile 在内存中由跨行距片段组成，并非连续 8 KiB panel。

```text
B_storage 的 N×K 视图（N 向下，K 向右）：
n=0: [k=0..31][k=32..63][...]
n=1: [k=0..31][k=32..63][...]
 ...
n=127: [...]
n=128: [...]

一次 B0 load：当前 N0 的 128 行，各取当前 K tile 的 64 B。
一次 B1 load：在 B0 起点基础上加 128*K 个 BF16，再取对应的 128 行。
```

在上述 N×K 存储视图中，Z 表示逐行（K 先变），N 表示逐列（N 先变）。当前是普通 Z 行主序；对应数学 W 的 K×N 视图则为列主序。以 tile 索引完全展开，实际地址次序是 `[Ntile][Nwithin][Ktile][Kwithin]`，而不是 tile-contiguous 的 `[Ntile][Ktile][Nwithin][Kwithin]`。所以不能仅凭“外 Z 内 Z”或“外 N 内 Z”推断当前布局。

QKᵀ 的右操作数转置视图自然对应每个 key token 的连续 head-dimension。AV 若输入 V 按 `[K,N]` 行主序提供，则需要在进入本 kernel 前形成 `[N,K]` 加载视图。转置/packing 成本单独计量。

若以后探索 tile-contiguous 或 N-pair 打包，应同时调整 B 地址生成和 stride，并重新测量缓存行为；本算例保留归档已测的 `[N,K]` 布局。

## Full tile 与原子发射顺序

采用 `M=128,N=128,K=32` full tile：当前 512-bit 行容纳 32 个 BF16，每条 MMACC 完成 `128*128*32` MAC。32 是本算例 full-tile 调度的 K 步长，不代表 ISA 的所有部分 tile 都必须以 32 递增。

外层按 M tile、N pair 遍历；一个 block 在完整 K reduction 内持有一对输出。寄存器映射为：

| 对象 | 偶数 K tile | 奇数 K tile |
| --- | --- | --- |
| A | tr0 | tr1 |
| B0 | tr2 | tr3 |
| B1 | tr3 | tr2 |
| C0/C1 | acc0/acc1 或 acc2/acc3 | 同一 pair 持续归约 |

每个 block 配置 tile 并清零当前 CReg pair，初始按 A0→tr0、B0→tr2、B1→tr3 加载。令 `c=0` 或 `2`，稳定阶段的发射顺序如下：

| 次序 | 偶数 K tile | 奇数 K tile |
| --- | --- | --- |
| 1 | `mmacc acc(c), tr2, tr0`（N0） | `mmacc acc(c), tr3, tr1`（N0） |
| 2 | 下一 K 的 B1 → tr2 | 下一 K 的 B1 → tr3 |
| 3 | `mmacc acc(c+1), tr3, tr0`（N1） | `mmacc acc(c+1), tr2, tr1`（N1） |
| 4 | 下一 K 的 B0 → tr3 | 下一 K 的 B0 → tr2 |
| 5 | 下一 K 的 A → tr1 | 下一 K 的 A → tr0 |
| 6 | 检查前一 block 的延迟 store | 检查前一 block 的延迟 store |

N0-first 释放的 BReg 先接收下一 K 的 N1，后者距离消费者跨越当前 N1 和下一 N0 的计算，提供预取窗口。BReg 的所有权随 K 奇偶翻转；C0/C1 与 N0/N1 的对应关系始终不变。

singleton N（包括 AV 的 N=128）每 K 只有一条 MMACC；tr2/tr3 用于交替载入当前/后继 B0，仅清零和写回 C0。它仍按 M tile 逐个执行，不能将 tr0/tr1 改解释成两个 M tile。

```text
block j:   [zero pair p][A/B warm-up][complete K reduction]──pending
block j+1:             [zero pair p^2][K1][K2]...[Klast]
                                       │   │
previous C pair:                    store0 store1
kernel end: flush last pending pair → mrelease → macquire → end counters
```

第 1/2 个 K tile 的计算发射后尝试写回前一 block 的 C0/C1；这是发射顺序，不表示软件等待计算完成。短 K 在 block 末尾补齐旧 pending stores，登记当前 pair，再由下一 block 或 kernel 尾部写回。硬件依赖跟踪保证 BReg 覆盖及 CReg 清零不会越过相关读取；完整 K reduction 中每个输出只进行一次最终 store。

理想连续计算要求 A/B 供给能覆盖两条 MMACC 的消费节奏，并让另一 CReg pair 的 store 与当前计算重叠。BF16 两个输入同为 16 bit，访存压力高于 packed i2；QKᵀ 的短 K 容易受输出写回影响，AV 的 singleton N 降低 A reuse。上述策略是这两个归档形状共同采用的已测方案，不据此推定任意新形状都达到相同利用率。

## 计时与验证口径

初始化、全局矩阵类型配置、sync reset 和 initial `mfence` 在计时之前。区间覆盖 block 内 tile 配置、mzero、A/B load、MMACC、scalar issue、全部 C-store，以及唯一 `mrelease/macquire`。归档 BF16 实现由 `macquire` 完成最终写回等待后读取计数器，本算例保留这一序列。

```text
MMACC tasks = (M/128)*(N/128)*(K/32)
C stores    = (M/128)*(N/128)
operations  = 2*M*N*K
```

输出 `utilization_bp` 使用有效 operations 除以 `kernel_cycles*2048`，它表示相对 BF16 结构峰值的有效计算比例。若用 ChiselDB 统计 MTE 生命周期，则 `MTE active/kernel_cycles` 是事件占用率，两者分母和工作量口径应分别说明。

逐 tile 补充检查放在原测量流程后的独立函数中，保持测量代码的寄存器分配。迁移验证比较两个原始尺寸从初始化到结束计数器的汇编，包括 scalar 控制流、地址计算和矩阵指令；小尺寸 NEMU 检查覆盖 singleton N、奇数 M/K tile、短 K 收尾和完整输出校验。

### 本次验证记录（2026-09-13）

使用 LLVM `064e86c4`、NEMU feat-bitnet `bc8f833fc`。QKᵀ 和 AV 均构建通过；归档与新代码在同一工具链下，从 main 入口至内核结束的最后一次 `rdinstret`，符号地址下的汇编序列、对象文件指令编码和重定位表达式均一致（局部基本块标签编号归一化）。新增逐 tile 校验位于此区间之后。

| NEMU [M,N,K] | 校验 | MMACC 数 | C-store 数 | 结果 |
| --- | --- | ---: | ---: | --- |
| [128,128,32] | 全输出 | 1 | 1 | PASS / GOOD TRAP |
| [128,256,32] | 逐 tile | 2 | 2 | PASS / GOOD TRAP |
| [384,384,96] | 逐 tile | 27 | 9 | PASS / GOOD TRAP |

MMACC/C-store 数按调度与程序报告核对；六种非法尺寸或校验模式组合均在编译时拒绝。完整尺寸性能沿用归档调度证据，本次迁移使用构建、代码生成比对和小尺寸 NEMU 验证。
