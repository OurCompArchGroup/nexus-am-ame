MARCH ?= rv64gc

include $(AM_HOME)/am/arch/isa/riscv64.mk

# Sail-targeted runtime: HTIF console + HTIF exit, no UART-lite / 16550 MMIO
# and no NEMU-trap halt. The boot/linker layout is shared with the nemu/xs
# runtimes, which already base the image at 0x80000000 (Sail's default RAM).
AM_SRCS := sail/isa/riscv/trm.c \
           sail/common/htif.c \
           nemu/common/mainargs.S \
           nemu/isa/riscv/boot/start.S

# Image is based at 0x80000000, so symbols are not reachable with the default
# (medlow) code model; force medany for both the GNU and LLVM toolchains.
CFLAGS  += -mcmodel=medany
ASFLAGS += -mcmodel=medany

CFLAGS  += -I$(AM_HOME)/am/src/nemu/include -I$(AM_HOME)/am/src/xs/include -DISA_H=\"riscv.h\"

ASFLAGS += -DMAINARGS=\"$(mainargs)\"
.PHONY: $(AM_HOME)/am/src/nemu/common/mainargs.S

LDFLAGS += -L $(AM_HOME)/am/src/nemu/ldscript
LDFLAGS += -T $(AM_HOME)/am/src/nemu/isa/riscv/boot/loader64.ld

image:
	@echo + LD "->" $(BINARY_REL).elf
	@$(LD) $(LDFLAGS) --gc-sections -o $(BINARY).elf --start-group $(LINK_FILES) --end-group
	@$(OBJDUMP) $(OBJDUMP_FLAGS) -d $(BINARY).elf > $(BINARY).txt
	@echo + OBJCOPY "->" $(BINARY_REL).bin
	@$(OBJCOPY) -S --set-section-flags .bss=alloc,contents -O binary $(BINARY).elf $(BINARY).bin

# Run under Sail. Point SAIL_RISCV at the emulator built from the sail repo,
# e.g. make ARCH=riscv64-sail run SAIL_RISCV=/path/to/riscv_sim_rv64
run:
	$(SAIL_RISCV) $(BINARY).elf
