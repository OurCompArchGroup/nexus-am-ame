---
name: xs-testcase-to-sail
description: Port a XiangShan (nexus-am) bare-metal test app to run on the Sail RISC-V emulator using the riscv64-sail AM target — HTIF console + HTIF exit instead of UART-lite / NEMU-trap halt. Use when a nexus-am app under apps/ must run on Sail, when `make ARCH=riscv64-xs` output faults on 0x40600000 under Sail, or when a link against uptime/_ioe_init/UART-lite must be removed for the minimal Sail runtime.
---

# Port a XiangShan test app to Sail

Sail runs a normal RV64 ELF but only routes MMIO for CLINT, HTIF, and signature
memory. The `riscv64-xs` runtime uses UART-lite at `0x40600000` and halts via a
NEMU-trap custom instruction — neither exists in Sail, so the `xs` image faults
during startup. The fix is a runtime swap, not an app rewrite: use the
`riscv64-sail` AM target, which keeps the app and the NEMU boot/linker layout
(already based at `0x80000000`, Sail's default RAM) and replaces only the
console and halt paths with HTIF.

Full background: `docs/xs-testcase-adaptation.md`.

## The runtime target (already in-tree)

`riscv64-sail` is implemented and does not need re-creating. Its pieces:

- `am/arch/riscv64-sail.mk` — sources `sail/isa/riscv/trm.c` + `sail/common/htif.c`
  plus shared NEMU `mainargs.S`/`start.S`; forces `-mcmodel=medany` (medlow can't
  reach `0x80000000`); reuses NEMU `loader64.ld`; adds a `run:` rule.
- `am/src/sail/common/htif.c` — `volatile uint64_t tohost, fromhost` (aligned 8;
  Sail enables HTIF only when it finds `tohost`), `__am_htif_putchar` (device 1
  cmd 1: `(1<<56)|(1<<48)|byte`, spin until `tohost`==0), `__am_htif_exit`
  (`tohost = (code<<1)|1`).
- `am/src/sail/isa/riscv/trm.c` — `_putc` → HTIF putchar, `_halt` → HTIF exit,
  standard `_heap` + `_trm_init`.

## Porting a new app

1. Build with the target — most apps need zero source changes:
   ```sh
   make ARCH=riscv64-sail
   ```
2. If the link fails on TRM services the minimal Sail runtime omits
   (`uptime`, `_ioe_init`, UART-lite helpers, timer/IOE APIs), remove or stub
   those call sites in the app. They are wall-clock/timing conveniences, not
   correctness. `csr_read(CSR_MCYCLE)` / `csr_read(CSR_MINSTRET)` still work for
   cycle/instret counts.
3. Lower any huge iteration count so the emulation finishes quickly.
4. Run:
   ```sh
   make ARCH=riscv64-sail run SAIL_RISCV=/path/to/riscv_sim_rv64
   ```
   Output prints via HTIF; the app exits via HTIF with `main`'s return code.

## Worked examples

- `apps/hello` — no source change; `make ARCH=riscv64-sail` and run.
- `apps/dhrystone` — commented out `Start_Timer`/`Stop_Timer`,
  `Begin_Time`/`End_Time`/`User_Time`, and the `_ioe_init()` call (all depend on
  `uptime()`), and lowered `NUMBER_OF_RUNS` (500000 → 5000). Diff `dry.c` to see
  the exact pattern.

## Do not

- Do not run the `riscv64-xs` binary unchanged on Sail (faults on `0x40600000`).
- Do not add a UART device to Sail just to keep the old console path.
- Do not rely on `printf`/console unless it routes through `_putc` → HTIF.
- Do not keep medlow: symbols at `0x80000000` are unreachable without medany.

## Checklist

- [ ] `make ARCH=riscv64-sail` builds `build/<name>-riscv64-sail.elf`.
- [ ] No unresolved link refs to `uptime`/`_ioe_init`/UART-lite.
- [ ] ELF is RV64, based at `0x80000000`, and contains `tohost`.
- [ ] `run` under Sail prints expected output and exits with the right code.
