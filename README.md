# The Abstract Machine (AM)

This repository includes test cases for the Accelerator Matrix Extension (AME).

## Get Started for Memory Images (Workloads)

In this section, we demonstrate how to build memory images (`base_address = 0x80000000`) for simulation.

> **Note:** Only `riscv64-xs` architecture has been tested. Other architectures may not work correctly.

- clone `nexus-am` from github:

```shell
git clone https://github.com/OpenXiangShan/nexus-am.git
cd nexus-am
export AM_HOME=`pwd`  # set AM_HOME
```

### Building AME Test Cases

```shell
cd apps/<test-case-name>
make ARCH=riscv64-xs TOOLCHAIN=LLVM
```

After building, the following files will be generated in `./build/`:

- `<name>-riscv64-xs.bin` - Memory image for RTL simulation
- `<name>-riscv64-xs.txt` - Objdump disassembly output
- `<name>-riscv64-xs.elf` - ELF executable

### Example

```shell
cd apps/ame-mmacc
make ARCH=riscv64-xs TOOLCHAIN=LLVM
ls ./build/
# Output: ame-mmacc-riscv64-xs.bin  ame-mmacc-riscv64-xs.txt  ame-mmacc-riscv64-xs.elf
```

Some AME bare-metal tests also support a runtime-selectable `64` mode through
AM `mainargs`. For those tests, build the `64` mode like this:

```shell
cd apps/ame-mmacc
make ARCH=riscv64-xs TOOLCHAIN=LLVM mainargs=64
```

Use `mainargs=128` or omit `mainargs` to build the default `128` mode.

### Running AME Test Cases

One can run the test cases using NEMU or XSAI.

```shell
# Run with NEMU
$NEMU_HOME/build/riscv64-nemu-interpreter -b build/ame-mmacc-riscv64-xs.bin

# Run with XSAI
$NOOP_HOME/build/emu -i ./build/ame-mmacc-riscv64-xs.bin
```

For NEMU or XSAI setup, please refer to [xsai-env documentation](https://github.com/Gs-ygc/xsai-env).

### Available AME Test Cases

| Test Case               | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `ame-ls-word`           | Matrix load/store operations with word-level (32-bit) elements |
| `ame-ls-ab`             | Matrix load operations for A/B matrices with transpose support |
| `ame-ls-whole`          | Whole-register matrix loads and stores                         |
| `ame-zero-word`         | Matrix accumulator zeroing operations (word-level)             |
| `ame-mmacc`             | Integer matrix multiply-accumulate operations                  |
| `ame-gemm`              | Matrix GEMM with a numerical self-check                        |
| `ame-mfence-mcfg`       | Matrix fence and configuration register checks                 |
| `ame-bf16`              | Scalar and vector BF16 operations                              |

---

For more information about multi-processor bring-up drivers, FPGA support, flash images, and other features, please refer to [ORIGNIAL_README.md](./ORIGNIAL_README.md).
