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
cd tests/<test-case-name>
make ARCH=riscv64-xs TOOLCHAIN=LLVM
```

After building, the following files will be generated in `./build/`:

- `<name>-riscv64-xs.bin` - Memory image for RTL simulation
- `<name>-riscv64-xs.txt` - Objdump disassembly output
- `<name>-riscv64-xs.elf` - ELF executable

### Example

```shell
cd tests/ame-mmacc
make ARCH=riscv64-xs TOOLCHAIN=LLVM
ls ./build/
# Output: ame-mmacc-riscv64-xs.bin  ame-mmacc-riscv64-xs.txt  ame-mmacc-riscv64-xs.elf
```

Some AME bare-metal tests also support a runtime-selectable `64` mode through
AM `mainargs`. For those tests, build the `64` mode like this:

```shell
cd tests/ame-mmacc
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

FP8 matrix tests require an implementation that advertises the corresponding
Zames `fp8e4m3/fp8e5m2 -> fp32` capability. Other tests should likewise be run
against an implementation that supports the types they configure through
`msetcfg`.

| Test Case               | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `tests/ame-gemm`                    | Deterministic multi-tile integer GEMM                       |
| `tests/ame-gemm-fuzz`               | Randomized integer GEMM with a scalar reference             |
| `tests/ame-ls-word`                 | Matrix load/store with word-level (32-bit) elements         |
| `tests/ame-ls-ab`                   | A/B matrix loads with transpose support                     |
| `tests/ame-ls-whole`                | Proposal-14 whole-register load/store operations            |
| `tests/ame-ls-whole-no-tile-store`  | Whole-register loads and C stores without tile stores       |
| `tests/ame-zero-word`               | Matrix accumulator and tile zeroing operations              |
| `tests/ame-mmacc`                   | Integer matrix multiply-accumulate operations               |
| `tests/ame-mfence-mcfg`             | `msetcfg/mgetcfg` and Zmasync `mfence` behavior             |
| `tests/ame-mmacc-fp8e4m3-fp32`      | FP8 E4M3 matrix multiply-accumulate into FP32               |
| `tests/ame-mmacc-fp8e5m2-fp32`      | FP8 E5M2 matrix multiply-accumulate into FP32               |
| `apps/ame-transpose-ab`             | Targeted A/B transpose-load precision probes               |
| `apps/ame-gemm-blas`                | XSAI BLAS high-level GEMM integration                       |

---

For more information about multi-processor bring-up drivers, FPGA support, flash images, and other features, please refer to [ORIGNIAL_README.md](./ORIGNIAL_README.md).
