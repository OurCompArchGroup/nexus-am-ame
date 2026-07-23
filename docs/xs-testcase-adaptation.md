# Adapting XiangShan Bare-Metal Testcases For Sail

This note captures the practical changes needed to run XiangShan `xs` bare-metal
testcases under the Sail RISC-V emulator in this repo.

## What Sail Expects

Sail will execute a normal RISC-V ELF, but the binary must match the platform
model selected by the CMake config.

Key points:

- The ELF must match the model width, i.e. RV32 ELF on RV32 Sail, RV64 ELF on RV64 Sail.
- The image must be statically linked and load into a configured memory region.
- Sail's default test/config path expects RAM at `0x80000000`.
- Sail's platform model only routes MMIO for CLINT, HTIF, and signature memory.
- Sail enables HTIF only if the main ELF provides a `tohost` symbol.

Relevant code:

- [c_emulator/riscv_sim.cpp](../c_emulator/riscv_sim.cpp) loads the ELF, scans for `tohost`, and enables HTIF if present.
- [model/sys/platform.sail](../model/sys/platform.sail) implements HTIF and CLINT MMIO.
- [config/config.json.in](../config/config.json.in) defines the default memory map.

## Why The Current XiangShan `xs` Apps Fail

The XiangShan AM runtime used by `riscv64-xs` is tied to UART-lite. That is
fine for XiangShan RTL simulation, but Sail does not implement that device.

In the current tree:

- `__am_init_uartlite()` and `__am_uartlite_putchar()` are used by the `xs` runtime.
- The UART-lite base is hardcoded to `0x40600000`.
- The produced ELF writes to `0x4060000c` during startup.

That address is not part of Sail's default MMIO device set, so the simulator
hits an access fault and the trap loop detector eventually reports failure.

Evidence in the repo:

- [am/src/nutshell/common/uartlite.c](https://github.com/OpenXiangShan/nexus-am/blob/main/am/src/nutshell/common/uartlite.c) hardcodes the UART-lite MMIO base.
- [am/src/noop/isa/riscv/trm.c](https://github.com/OpenXiangShan/nexus-am/blob/main/am/src/noop/isa/riscv/trm.c) shows the startup shape: init serial, call `main`, then halt.
- The Sail model's MMIO dispatch is limited to CLINT, signature memory, and HTIF in [model/sys/platform.sail](../model/sys/platform.sail).

## Minimal Adaptation Strategy

The least invasive path is to add a Sail-targeted runtime variant instead of
reusing the current `riscv64-xs` image.

Recommended changes:

1. Add a Sail-specific AM platform or runtime variant.
   - Keep the app code unchanged.
   - Replace UART-lite console I/O with HTIF console I/O.
   - Replace the current trap-based halt path with an HTIF exit path.

2. Emit `tohost` and `fromhost` symbols in the ELF.
   - Sail looks for `tohost` in the main ELF and enables HTIF only if it exists.
   - The compliance-style pattern in [tests/crypto/include/model_test.h](../tests/crypto/include/model_test.h) is the relevant shape.

3. Keep the linker layout compatible with Sail's default memory map.
   - Base text/data at `0x80000000`.
   - Avoid device accesses outside the configured MMIO window.

4. Avoid device-specific assumptions in startup code.
   - Do not call UART-lite init.
   - Do not busy-wait on UART-lite status registers.
   - Prefer a simple `putc` path that writes through HTIF.

## Implemented Adaptation (`riscv64-sail`)

The strategy above was implemented as a new AM arch target, `riscv64-sail`.
`apps/hello` and `apps/dhrystone` both build and run under Sail with it. The
target reuses the existing NEMU boot/linker layout (already based at
`0x80000000`) and swaps only the console and halt paths.

### New runtime files

- [am/arch/riscv64-sail.mk](../am/arch/riscv64-sail.mk) — the arch selection.
  - Sources: `sail/isa/riscv/trm.c`, `sail/common/htif.c`, plus the shared
    NEMU `mainargs.S` and `start.S`.
  - `-mcmodel=medany` for both CC and AS (medlow can't reach `0x80000000`).
  - Reuses the NEMU `loader64.ld` linker script and ldscript dir.
  - Adds a `run:` rule invoking `$(SAIL_RISCV) $(BINARY).elf`.
- [am/src/sail/common/htif.c](../am/src/sail/common/htif.c) — HTIF mailbox.
  - Declares `volatile uint64_t tohost, fromhost` (aligned 8) so Sail detects
    `tohost` and enables its HTIF device; both live in loadable RAM.
  - `__am_htif_putchar()`: console write is device 1, command 1
    (`(1<<56)|(1<<48)|byte`), then spin until Sail clears `tohost`.
  - `__am_htif_exit()`: `tohost = (code<<1)|1` (bit 0 = halt).
- [am/src/sail/isa/riscv/trm.c](../am/src/sail/isa/riscv/trm.c) — TRM shim.
  - `_putc()` routes through `__am_htif_putchar`.
  - `_halt()` routes through `__am_htif_exit` (not the NEMU-trap custom
    instruction `.word 0x0005006b`, which Sail does not implement).
  - `_heap` + `_trm_init()` are the standard shape.

### App-level notes

- `apps/hello` needed **no** source change — just `make ARCH=riscv64-sail`.
- `apps/dhrystone` needed the timer removed: `uptime()` / `_ioe_init()` are not
  provided by the minimal Sail TRM, so `Start_Timer`/`Stop_Timer` and the
  `Begin_Time`/`End_Time`/`User_Time` bookkeeping were commented out, and
  `NUMBER_OF_RUNS` was lowered (500000 → 5000) to keep the emulation short.
  Cycle/instret via `csr_read(CSR_MCYCLE/MINSTRET)` still work.

### Build & run

```sh
make ARCH=riscv64-sail                       # build build/<name>-riscv64-sail.elf
make ARCH=riscv64-sail run SAIL_RISCV=/path/to/riscv_sim_rv64
```

## What To Change In The XiangShan Tree

If you are adapting `nexus-am`, the concrete places to touch are:

- `am/src/nutshell/common/uartlite.c`
- `am/src/noop/isa/riscv/trm.c`
- The relevant `am/arch/*.mk` selection for the target
- Any startup/halting assembly or C that assumes UART-lite MMIO

Conceptually:

- Keep the application code.
- Swap the platform support layer.
- Make the runtime speak HTIF instead of UART-lite.

## What Not To Do

- Do not try to run the current `riscv64-xs` binary unchanged under Sail.
- Do not expect Sail to emulate `0x40600000` unless you add a new MMIO device.
- Do not rely on `printf` output unless the runtime routes it to Sail-supported I/O.

## Quick Check List

Before trying a testcase on Sail, confirm:

- ELF class matches the Sail model width.
- The ELF loads at `0x80000000` or another configured RAM region.
- The ELF contains `tohost` if you want Sail HTIF output and exit handling.
- The runtime does not touch UART-lite or 16550 registers.
- The application halts via HTIF, not via a device-specific trap path.

For a new app, this usually reduces to: build with `ARCH=riscv64-sail`, and if
the link fails on `uptime`/`_ioe_init`/other TRM services the minimal Sail
runtime omits, stub or remove those call sites (see `apps/dhrystone`).

## Bottom Line

For XiangShan bare-metal testcases, the adaptation target is not the app logic;
it is the runtime shim underneath it. The Sail-friendly shape is:

- same application
- different startup/runtime layer
- HTIF-backed console and exit
- no UART-lite dependency

