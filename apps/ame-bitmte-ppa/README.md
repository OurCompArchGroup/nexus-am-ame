# 六模式裸机 GEMM 性能与功耗负载

本算例计算 `C[M,N] = A[M,K] × B_storage[N,K]^T`。六种精度共享整数数值输入，A 为 `[-8,7]`，B 为 `{0,1}`，C 初值为零；固定 seed 后，仅物理位宽和调度变化。`npair`、`mpair`、`npair-fixed` 与 `quad-roll` 是通过实际 RTL 事件比较的调度候选。

| MODE | A / B / C | 单条 MMACC 的 K | 一次 B load 的驻留 K | kslice |
| --- | --- | ---: | ---: | --- |
| i8-i8 | i8 / i8 / i32 | 64 | 64 | 0 |
| bf16-bf16 | BF16 / BF16 / FP32 | 32 | 32 | 0 |
| i8-fp2pack4 | i8 / ternary / i32 | 64 | 256 | 0..3 |
| i4-fp2pack4 | i4 / ternary / i32 | 128 | 256 | 0..1 |
| i8-i1pack8 | i8 / 0,1 / i32 | 64 | 512 | 0..7 |
| i4-i1pack8 | i4 / 0,1 / i32 | 128 | 512 | 0..3 |

M、N 为正的 128 倍数，公共实验 K 为正的 512 倍数。FP32 参考要求 K 不超过 2,097,152，使任意部分和始终落在可精确表示的整数域。i4 采用低 nibble 对应较低 K，ternary 使用 `00=0,01=+1` 的公共数据子集，binary 为 `0=0,1=1`。通用 ternary 的 `11=-1` 和 i4 全范围语义见已有 `apps/ame-bitmte`；本实验公共数据用于六模式可比测量。

A、B 均按行主序存储，B 的一个存储行对应一个输出列，K 连续；三个矩阵 64B 对齐。A/B 的行 stride 分别为 `K*A_BITS/8` 与 `K*B_BITS/8`，C stride 为 `4N`。所有 typed A/B panel load 每行读取 64B，共读取 128 行，即 8KiB。packed B 保持原生表示，计算用 selector 选择其驻留行的分片。

## 构建

从 xsai-env 根目录加载环境，使用既有 AM 构建规则：

```bash
source env.sh
make -C nexus-am/apps/ame-bitmte-ppa image \
  MODE=i8-fp2pack4 SCHEDULE=npair M=256 N=256 K=512 SEED=0x20261103
```

默认 `CHECK=0`，适合功耗负载：完成 GEMM 和最终 store 排空后退出，不遍历或校验结果。需要逐元素检查全部输出时显式传入 `CHECK=1`：

```bash
make -C nexus-am/apps/ame-bitmte-ppa image \
  MODE=i8-fp2pack4 SCHEDULE=mpair M=256 N=512 K=1024 CHECK=1
```

`CHECK` 只接受 `0` 或 `1`，不改变 GEMM 调度、load/compute/store 或计时范围。替换 MODE 或 SCHEDULE 构建另一独立程序。默认生成 `build/<MODE>/<SCHEDULE>/<M>x<N>x<K>-s<SEED>-check<CHECK>/gemm.{elf,bin,txt}`，对象目录也包含所有参数；两个开关状态各自构建，不覆盖旧的无 check 后缀产物。`OUTPUT_ROOT=<绝对路径>` 可以将本次实验产物放入已有顶层 build 目录。每个 bin 只运行一个模式、一个尺寸、一次内核；程序不执行 kernel 预热或重复采样。当前 V3 在构建时生成完整 packed A/B、全部参考结果，以及显式 `.data` 零 C；参考结果数组仅在 `CHECK=1` 的 ELF/bin 中保留。既有 raw-bin loader 将镜像字节装入物理内存，guest 不再生成、packing、清 C 或遍历输入求 hash。启动、UART 打印、配置和 stack 仍在计时前执行，因此不宣称 ROI 开始时每个 cache line 都是 cold。构建时 Python 标准库生成含全部 fixture 的 `expected.h` 与带 SHA256 的 `expected.json`；两者随同 ELF/bin 绑定和保存。V1/V2 的 guest prepare 方式及其旧产物保留，性能证据使用当前明确绑定的数据准备版本。

## 调度候选

`npair` 逐个 M tile、N pair 遍历：tr0/tr1 在 K 维 ping-pong A，tr2/tr3 保存两个 N panel 的 packed B。每份 A 供两个输出列 tile 使用；一行 B 的所有 selector 消费完后，原 N0 寄存器预取下一个 resident 段的 N1，原 N1 寄存器预取下一个 N0，随后交换 B 所有权。此策略减少 A loader 的服务压力。

`npair-fixed` 保持npair的A双缓冲、输出pair、ACC与store指令顺序；完整N pair中固定TR2=N0、TR3=N1，每列在最后selector之后换入自身下一resident段。singleton尾组仍采用原B双缓冲。它只改变驻留段间B寄存器/列映射，任务与DMA字节数同npair；性能需独立真实RTL测量。

`mpair` 逐个 M pair、N tile 遍历：tr0/tr1 保存两组 A 行，tr2/tr3 在 packed B resident 段之间双缓冲。每份 B 供两个 M tile 使用，空闲 B 寄存器提前预取下一个 resident 段；A 的下一分片在其消费者之后排队载入。此策略减少 B loader 的服务压力，代价是更多 A load。

`mpair-bdefer` 仅为普通 i8-i8/BF16、C0=`256×512×512`、seed=`0x20261103` 的独立有限挑战者。它在完整 M pair 的每个非最终 K resident 中，将下一 B 寄存器的预载排到当前第二行 MMACC 后、下一 A1 load 前；旧 `mpair` 保留不变。矩阵任务、地址与字节数不应改变，但该顺序可能影响异步窗口与后续 B 就绪，需独立软件及真实 RTL 证据，不能预先宣称更快。

`quad-roll` 以最多2M×2N输出组成一组，TR0/1保存两行A、TR2/3保存两列B，ACC0/1/2/3保存C00/C01/C10/C11。每组首Kslot只seed一次；A在最后行消费者之后换入，B在resident最后selector的最后列消费者之后换入。末slot逐ACC排队store，按紧凑尾序提前准备下一组A0/B0，旧store全部写响应完成后各ACC独立zero和seed；row1/col1守卫处理单行/单列尾。它复用两份A和两份B，性能仍须真实RTL测量。

这些候选均使用 acc0/acc1 与 acc2/acc3 交替保存输出组。每个输出从 mzero 开始，完整 K reduction 留在 acc，最终只写回一次。前一输出组的两条 store 在下一组第 1/2 个计算 pair 发出后排队；这是发射顺序，完成与寄存器复用由真实硬件依赖跟踪保证。短组在结尾补发尚未发出的旧 store。整个 kernel 只有一次最终 release/acquire；不在 K 维进行中间 store 或同步。

`npair-fixed`的命令计数与下表npair列相同。`quad-roll`的Aload采用npair公式，Bload采用mpair公式；MMACC、mzero、最终store和同步计数不变。quad的TR/ACC角色固定，不按pair轮换输出组；各ACC按旧store完成独立复用。

令 `mt=M/128`、`nt=N/128`、`q=K/Kslot`、`r=K/Kresident`：

| 命令 | npair | mpair |
| --- | ---: | ---: |
| MMACC | mt×nt×q | mt×nt×q |
| A load | mt×ceil(nt/2)×q | mt×nt×q |
| B load | mt×nt×r | ceil(mt/2)×nt×r |
| mzero / 最终 C store | 各 mt×nt | 各 mt×nt |
| 显式 C load | 0 | 0 |

矩阵 DMA payload 为 `8192*(A_load+B_load)+65536*C_store` 字节；它是命令传输量，实际 cache refill 流量需单独观测。

## 计时与校验

输入准备和 packing 在 host 构建期完成；矩阵类型/shape 配置、初始 sync reset/mfence 和打印在 guest 计时之前。`ppa_roi_begin` / `ppa_roi_end` 为 ELF 中的全局符号。`rdcycle` 覆盖首个 mzero 发射、全部 load/MMACC/最终 store、标量循环与寄存器选择，直到最终 release/acquire 和 mfence 完成。构建时由 `oracle.py` 独立生成全部 M×N 个整数参考结果，使用 A 的四位补码 bit-plane 与 binary B 的交集 popcount 得到精确点积，不依赖 kernel 分块、packing 或 selector。首个公共尺寸已与独立标量三循环逐元素交叉验证。仅在 `CHECK=1` 时，guest 在计时后逐个比较全部 M×N 个输出，时间复杂度为 O(MN)；BF16 按整数参考的精确 FP32 位型校验。默认 `CHECK=0` 编译时移除 guest 校验路径，host 仍生成完整 fixture/oracle，便于独立保存输入身份。

`PPA_BEGIN` 输出模式、调度、MNK、seed、公共输入 hash、`input_hash_source=host_fixture`、分片几何和 `check` 状态。输入 hash 是 host fixture 的数值身份，未声称由 guest 动态遍历得到；实际 ELF/bin 的全部 packed lane 和字节由独立 host 审计核对。`CHECK=1` 全量校验通过后输出 `PPA_PASS ... check=1`，包含完整内核 cycles、有效运算数、期望命令计数、输入及数学结果 hash。BF16 的数学结果 hash 使用对应整数结果，因此六模式可直接核对；raw 输出仍按 FP32 位型检查。`CHECK=0` 完成后输出 `PPA_DONE ... check=0` 及同类计时/命令计数，不输出结果 hash；该标记只表示 kernel 已完成，不证明结果正确。

2026-11-03 用户更新了主计时口径：主性能为 DB 的 `memory_span`，从首个有效 matrix load issue 到最后一条 C store 的全部写响应完成；可用已核对完成语义的 release/acquire 作为排空证据。完整 guest `rdcycle` 与首个真实 MMACC issue 到最后一个 MMACC finish 的 compute span 均为辅助指标，保留各自内部全部空泡。原先以完整 guest kernel cycles 作为主指标的口径已失效，旧源码和证据保留，不改写旧结果。调度与尺寸平台的 2% 判断使用 `memory_span/(2MNK)`，不能用 NEMU cycles 或辅助窗口代替。阵列有效计算拍、任务生命周期和 L/C/S 重叠按真实事件分别报告。最终功耗交接可使用同一 bin，按已验证的 memory span 与任务事件限定采样区间。

公共起始点 256×256×512 只有两个输出 pair，尚无 ACC pair 返回复用；binary 的 mpair 此时只用 TR0/1/2。两个调度都完整成对且至少三组时，最小输出域为 256×512 或 512×256，具有四个完整 pair。寄存器曾出现不等于 load/compute/store 已充分 overlap，最终仍需真实 DB 的任务生命周期和重复 ACC 复用证据。
