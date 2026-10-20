# EW/Reduce 软件开发用正向 payload

本程序为 DefaultConfig（每个 acc 为 128×128 个32位槽、32个 LV，各128×32位）提供可运行的指令与算子范例。使用当前工作区扩展后的 Zames LLVM 和 NEMU；原 proposal14 编译器/模拟器无法支持这些新增指令。指令通过可读助记符和显式物理寄存器 clobber 使用。

测试覆盖 EW/Reduce/LV、格式转换及宽累加状态，并展示这些操作与访存、mmacc的组合用法。数值检查包括浮点特殊值、IEEE sticky flags、整数饱和/回绕和合法空维度的结果。

[ISA、实现与使用手册](SOFTWARE-HANDOFF.md) 列出 ISA 行为、机器编码、CSR/数值、兼容变化、NEMU/LLVM/spec 代码改动和工具/算子用法。

最终实测：**147 个正向场景、851909 项检查，GOODTRAP0，退出0**。身份与日志位于 xsai-env 的 `log/ew-reduce-software-20261021-01a15800/`。

## 构建和运行

从 xsai-env 根目录执行；显式指定路径，避免机器环境的 AM_HOME/LLVM_HOME 指向其他工作区或旧安装：

```sh
source env.sh
export AM_HOME="$XS_PROJECT_ROOT/nexus-am"
export LLVM_HOME="$XS_PROJECT_ROOT/llvm-project-ame/build"
export NEMU_HOME="$XS_PROJECT_ROOT/NEMU"

# 从当前源码构建所需工具；已有对应版本时可跳过。
cmake --build "$LLVM_HOME" --target clang llvm-mc llvm-objdump llvm-ar llvm-objcopy llvm-readelf lld --parallel 8

make -C "$AM_HOME/tests/ame-ew" ARCH=riscv64-xs TOOLCHAIN=LLVM \
  AM_HOME="$AM_HOME" LLVM_HOME="$LLVM_HOME"
make -C "$AM_HOME/tests/ame-ew" ARCH=riscv64-xs TOOLCHAIN=LLVM run-nemu
```

Makefile 复用 `nexus-am/Makefile.app`，使用 `-march=rv64gc_zames_zmasync`。`riscv64-xs` 使用当前板级 UART 映射，`run-nemu` 通过现有 xsai-env 根 Makefile 启动 NEMU。

生成物位于本目录 `build/`：

- `ame-ew-riscv64-xs.elf`：符号与调试信息。
- `ame-ew-riscv64-xs.bin`：可从 0x80000000 运行的真实裸机 payload。
- `ame-ew-riscv64-xs.txt`：启用 `+zames,+zmasync` 的反汇编。

NEMU需采用手册中的standalone矩阵配置。单独运行与反汇编示例：

```sh
"$NEMU_HOME/build/riscv64-nemu-interpreter" -b -I 100000000 \
  -l /tmp/ame-ew-nemu.log "$AM_HOME/tests/ame-ew/build/ame-ew-riscv64-xs.bin"
"$LLVM_HOME/bin/llvm-objdump" -d --mattr=+zames,+zmasync \
  "$AM_HOME/tests/ame-ew/build/ame-ew-riscv64-xs.elf"
```

成功终态为 `AME_EW_RESULT cases=... checks=... failures=0 signature=4557504f`、NEMU `HIT GOOD TRAP`、退出0。仅生成 ELF、输出部分 `[CASE]` 或超时不算通过。失败会打印 case、检查标签、源码行号和期望/实际位模式，然后以非零状态停止。

## 从范例开始开发

| 文件 | 内容 |
| --- | --- |
| `ame-ew.h` | 测试局部的显式寄存器 inline asm 宏、配置常量、形状和 flags 辅助；不是 intrinsic 或 ABI |
| `ew.c` | 逐元素运算、广播、ABS/SFU、类型与位模式示例 |
| `reduce.c` | 行/列 ADD/MAX、旧值累加、整数最终一次饱和及浮点规则 |
| `memory.c` | LV 访存、尾部、32寄存器、名义 C 内存格式和宽槽 |
| `convert.c` | 整 acc 转换、同类保留、跨类 FCVT、舍入和异常标志 |
| `operators.c` | 融合 epilogue、再量化、统计量、softmax2、自然指数softmax、FP16 mmacc 后处理 |
| `main.c` | 形状/状态初始化、断言、case/check 计数与终态 |

最容易混淆的几点：

- `op md, B, A`：`msub acc2, acc1, acc0` 是 `acc0-acc1`。
- EW `.row` 使用 `lv[j]`（长度 N）；EW `.col` 使用 `lv[i]`（长度 M）。Reduce `.row` 沿 N 归约，产生 M 个值，通常随后用 EW `.col` 广播。因此 `vredadd.row` 的结果可用 `vs.col` 写回 M 个值。
- `.x` 参数是 GPR 低32位。浮点参数用 `f32_bits(value)` 传递位模式，不能传整数数值1来表示浮点1.0。
- `msetcfg` 只改名义配置、保留原始位；`msetcfg.cvt` 才把全部16384个槽按旧/新类别转换，忽略活动 M/N。调用前要保证整个 acc 的源数据有效；只初始化小 tile 不代表其余槽已初始化。
- INT8/INT32 内部均 INT32；FP16/BF16/FP32 内部均 FP32。同类别改名义精度不立刻舍入；普通 C store 才窄化。跨整数/浮点类别请显式转换。
- 矩阵 rounding/flags 为 `mfrm`/`mfflags`，与标量 `frm`/`fflags` 独立。REC7 返回 FP32 位格式的7位倒数估计；需要更高精度可参照 `operators.c` 显式做 Newton 步骤。
- `operators.c` 分别演示 base-2 softmax 和自然指数 softmax。后者先将减去最大值后的 logits 乘 `log2(e)` 再 EXP2，并使用矩阵内 LV 广播和显式 Newton；两者采用各自独立参考和示例容差。行均值/方差范例可作为 normalization 的统计阶段。

修改算子时先保持显式寄存器分工、cfg 和活动尺寸，再替换数据/计算序列。CPU 与 LV 通过访存交换数据。矩阵寄存器的选择、跨函数保存和恢复由软件显式管理；异步执行、线程切换与中断环境需结合对应同步和上下文机制。此payload的运行证据覆盖默认配置的单hart NEMU功能。
