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

Note that all `mfmacc` test cases are only tested on NEMU since floating-point matrix multiply is currently not available in XSAI. 

| Test Case               | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `ame-ls_word`           | Matrix load/store operations with word-level (32-bit) elements |
| `ame-ls-ab`             | Matrix load operations for A/B matrices with transpose support |
| `ame-zero`              | Matrix accumulator zeroing operations (byte-level)             |
| `ame-zero_word`         | Matrix accumulator zeroing operations (word-level)             |
| `ame-mmacc`             | Integer matrix multiply-accumulate operations                  |
| `ame-mfmacc_f32tof32`   | Floating-point matrix multiply-accumulate (FP32 to FP32)       |
| `ame-mfmacc_f64tof64`   | Floating-point matrix multiply-accumulate (FP64 to FP64)       |
| `ame-mfmacc_f16tof32`   | Floating-point matrix multiply-accumulate (FP16 to FP32)       |
| `ame-mfmacc_f8e5tof32`  | Floating-point matrix multiply-accumulate (FP8 E5M2 to FP32)   |
| `ame-mfmacc_f8e4tof32`  | Floating-point matrix multiply-accumulate (FP8 E4M3 to FP32)   |
| `ame-mfmacc_f8e5tof16`  | Floating-point matrix multiply-accumulate (FP8 E5M2 to FP16)   |
| `ame-mfmacc_f8e4tof16`  | Floating-point matrix multiply-accumulate (FP8 E4M3 to FP16)   |

---

For more information about multi-processor bring-up drivers, FPGA support, flash images, and other features, please refer to [ORIGNIAL_README.md](./ORIGNIAL_README.md).
