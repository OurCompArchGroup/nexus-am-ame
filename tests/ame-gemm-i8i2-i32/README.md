# i8 × packed i2 → i32 GEMM

本算例实现 `C[m,n] = Σ_k A[m,k] × W[k,n]`。A 为 signed i8，W 为 signed two's-complement i2，C 为 i32；每字节打包四个 i2 的存储类型为 `fp2pack4`。采用 BitVLA OpenLLC 已测的 `1M × 2N` output-stationary 调度，M/N/K 在构建时配置。

## 构建与检查

在已配置自定义 LLVM 的 xsai-env 根目录执行 `source env.sh`。NEMU、XSAI 和 XSAI/CUTE 使用支持 `fp2pack4` 的 `feat-bitnet` 分支，矩阵指令通过 `zames/zmasync` 汇编。独立使用 nexus-am 时设置 `AM_HOME` 为仓库根目录、`LLVM_HOME` 为该 LLVM 安装目录。

```bash
source env.sh
make -C nexus-am/tests/ame-gemm-i8i2-i32 M=256 N=256 K=128
NEMU/build/riscv64-nemu-interpreter -b nexus-am/tests/ame-gemm-i8i2-i32/build/ame-gemm-i8i2-i32-m256-n256-k128-vf0-riscv64-xs.bin
```

| 参数 | 默认值 | 约束 |
| --- | ---: | --- |
| M | 256 | 正的 128 倍数 |
| N | 256 | 正的 128 倍数 |
| K | 128 | 正的 64 倍数 |
| VERIFY_FULL | 0 | 0：逐输出 tile 抽样及边界检查；1：检查全部输出 |

每种尺寸和校验模式使用独立对象目录及 payload 名。默认校验覆盖每个输出 tile；输入 deterministic，随 tile、行及 K lane 变化。`VERIFY_FULL=1` 在计时区间外执行标量 expected-value 检查，适用于小尺寸正确性验证。矩阵静态分配，配置尺寸还需满足运行平台的内存容量。

四个 BitVLA 形状共用这一份内核：

```bash
make -C nexus-am/tests/ame-gemm-i8i2-i32 M=1024 N=2560 K=2560
make -C nexus-am/tests/ame-gemm-i8i2-i32 M=1024 N=640  K=2560
make -C nexus-am/tests/ame-gemm-i8i2-i32 M=1024 N=6912 K=2560
make -C nexus-am/tests/ame-gemm-i8i2-i32 M=1024 N=2560 K=6912
```

## 数据布局与地址公式

区分数学右操作数 `W[K,N]` 和加载视图 `B_storage[N,K]`：`B_storage[n,k] = W[k,n]`。下面用存储视图的 N 为纵轴、K 为横轴；Z 表示逐行遍历（横轴先变），N 表示逐列遍历（纵轴先变）。若画成数学 W 的 K×N 坐标，同一遍历的 Z/N 名称会交换，因此地址公式是布局的精确定义。

A 为 `[M,K]` i8 行主序，元素地址偏移为 `m*K+k` 字节；C 为 `[M,N]` i32 行主序，偏移为 `4*(m*N+n)` 字节。A load stride 为完整 K，C-store stride 为 `4N`。

B 使用分组 panel 布局：

```text
[NtilePair][Ktile][NwithinPair][Nwithin][Kwithin / 4][packed lane]

完整 N pair：
  Ktile 0: N0 panel (2 KiB), N1 panel (2 KiB)
  Ktile 1: N0 panel (2 KiB), N1 panel (2 KiB)
  ...
最后一个 singleton N pair：
  Ktile 0: N0 panel (2 KiB)
  Ktile 1: N0 panel (2 KiB)
  ...
```

设 `nt=n/128`、`ni=n%128`、`kt=k/64`、`ki=k%64`、`q=nt/2`、`Nt=N/128`、`Kt=K/64`，均为整数除法。令 `h=2` 表示 `2q+1<Nt`，否则 `h=1`，则：

```text
panel_index = 2*q*Kt + h*kt + (nt % 2)
byte_offset = 2048*panel_index + 16*ni + ki/4
lane_shift  = 2*(ki % 4)
code        = (B_bytes[byte_offset] >> lane_shift) & 3
value       = code < 2 ? code : code - 4
```

低 K lane 放在字节低位：`00=0`、`01=+1`、`10=-2`、`11=-1`。B 共占 `NK/4` 字节。每 panel 是逻辑 `128×64`、物理 `128×16 B`，四个相邻 N row 组成一条 64 B line；B load 的 row stride 固定为 16 B。`matrix_b_fp2` 以 4 KiB 对齐，连续 panel 的页内偏移为 `0x000` 或 `0x800`。

以 N×K 存储视图看，pair 网格 `[NtilePair][Ktile]` 为外层 Z，panel 内 `[Nwithin][Kwithin]` 也为 Z，但 pair 内还包含 N0/N1 panel 的交错层。准确说法是上述五维 panel 布局，不能省略 `NwithinPair` 后按普通 `[N,K]` 大行距加载。模型权重先按该地址公式打包，打包时间在 kernel 计时之外。

## 原子调度与寄存器所有权

满 tile 为 `M=128,N=128,K=64`。K=64 来自当前 512-bit A 行容纳 64 个 i8；fp2 packed-B 加载契约也要求此 full tile。每条 MMACC 执行 `128*128*64` MAC。

一次 output block 仅消费一个 M tile、最多两个 N tile；两 M tile 的外层分组用于缓存复用，组内的两个 M tile 仍依次计算。

```text
for each two-M-tile group:
  for each N pair:
    for each valid M tile in the group:
      use C pair 0 or 2
      reduce the complete K dimension
      switch C pair
```

M 含奇数个 tile 时，最后一组只处理一个 M tile。每个 block 使用以下寄存器：

| 对象 | 寄存器 | 生命周期 |
| --- | --- | --- |
| A(k) | 偶数 k 用 tr0，奇数 k 用 tr1 | 供 N1、N0 两条 MMACC 消费 |
| B0/B1 | 偶数 k：tr2/tr3；奇数 k：tr3/tr2 | cross-B 预取后交换所有权 |
| C0/C1 | acc0/acc1 或 acc2/acc3 | 完整 K reduction 保留输出 |
| 前一 block 输出 | 另一 CReg pair | 在当前 block 中发出最终 store |

完整 N pair 先清零当前 CReg pair，依次加载 B0→tr2、B1→tr3、A0→tr0。令 `c` 为 0 或 2，K tile 从 0 编号，稳定阶段的原子发射顺序为：

| 次序 | 偶数 K tile | 奇数 K tile |
| --- | --- | --- |
| 1 | `mmacc acc(c+1), tr3, tr0`（N1） | `mmacc acc(c+1), tr2, tr1`（N1） |
| 2 | 下一 K 的 B0 → tr3 | 下一 K 的 B0 → tr2 |
| 3 | `mmacc acc(c), tr2, tr0`（N0） | `mmacc acc(c), tr3, tr1`（N0） |
| 4 | 下一 K 的 B1 → tr2 | 下一 K 的 B1 → tr3 |
| 5 | 下一 K 的 A → tr1 | 下一 K 的 A → tr0 |
| 6 | 检查前一 block 的延迟 store | 检查前一 block 的延迟 store |

最后一个 K tile 不发后继 load。N1-first 使后继 B panel 按 B0、B1 的物理地址顺序进入 B loader。奇偶路径和 CReg pair 均使用固定寄存器指令，保留已测完整 N pair 的专用代码生成。

singleton N 仅加载 B0 并计算 C0；tr2/tr3 交替预取连续 K 的 B0。例如 K/V 的第五个 N tile，不发第二条 B load、MMACC 或 C-store。

```text
block j:    [zero C pair p][B/A warm-up][complete K reduction]──pending C
block j+1:                    [zero C pair p^2][K1 ... K12 ... K24 ...]
                                                   │        │
previous pair stores:                           store C0  store C1
kernel end: flush last pending pair → mrelease → macquire → mfence
```

图中 K12/K24 表示后继 block 已发射 12/24 个 K tile 的计算，而非等待计算完成。寄存器相关性由硬件保证：覆盖 BReg 要等待先前读取，复用 CReg 的 mzero 要等待先前 store 的读取。每个 block 末尾补发旧 pending pair 尚未发出的 store，再将当前结果登记为 pending，因而短 K 同样正确。

理想流水依靠一条 A load 服务两条 MMACC、B cross-prefetch 和 CReg pair 写回重叠。两 M tile 分组提高 B 的缓存复用机会，但命令层仍为每个 M tile 发 B load；它是 output-stationary 加缓存复用，不是两 M 同时驻留的 B-register weight-stationary。小 N、短 K 和缓存供给延迟会限制实际占用率。

## 计时与验证口径

配置矩阵、初始化同步及 initial fence 在计时之前。计时覆盖 mzero、全部 A/B load、MMACC、scalar issue、所有最终 C-store、唯一 `mrelease/macquire` 和末尾 `mfence`；随后读取结束计数器并做软件校验。

```text
MMACC tasks = (M/128)*(N/128)*(K/64)
C stores    = (M/128)*(N/128)
operations  = 2*M*N*K
```

源代码中的形状初始化分为初始化与覆盖两个步骤，完整 N pair 使用显式奇偶宏；这些细节用于保持归档内核的编译结果。修改后应比较包含 scalar 地址计算、循环和同步的完整内核，不能只比较 MMACC 数。

迁移验证包括四个原始尺寸的构建和归档代码生成比较，以及最小 tile、singleton N、奇数 M/K tile、短 K 和第 12/24 个 K tile 写回边界的 NEMU 检查。历史 OpenLLC 测量代表这套调度在相应形状上的结果；其他 MNK 的性能需结合供给带宽、缓存和填充/排空开销分析。

### 本次验证记录（2026-09-13）

使用 LLVM `064e86c4`、NEMU feat-bitnet `bc8f833fc`。四个原始尺寸均构建通过；归档与新代码在同一工具链下，从 main 入口至内核结束的最后一次 `rdinstret`，符号地址下的汇编序列、对象文件指令编码和重定位表达式均一致（局部基本块标签编号归一化）。这覆盖了初始化、寄存器分配、地址计算、分支、矩阵指令及 completion。

| NEMU [M,N,K] | 校验 | MMACC 数 | C-store 数 | 结果 |
| --- | --- | ---: | ---: | --- |
| [128,128,64] | 全输出 | 1 | 1 | PASS / GOOD TRAP |
| [256,256,128] | 逐 tile | 8 | 4 | PASS / GOOD TRAP |
| [384,384,192] | 逐 tile | 27 | 9 | PASS / GOOD TRAP |
| [256,256,832] | 逐 tile | 52 | 4 | PASS / GOOD TRAP |
| [256,256,1600] | 逐 tile | 100 | 4 | PASS / GOOD TRAP |

MMACC/C-store 数按调度与程序报告核对。另以实际 `b_panel_index` 函数检查 54 种 N/K tile 网格的 panel 覆盖、singleton、页对齐和 lane 编码；六种非法参数组合均在编译时拒绝。完整尺寸 ELF 中的 `matrix_b_fp2` 均为 4 KiB 对齐，大小为 `NK/4`。
