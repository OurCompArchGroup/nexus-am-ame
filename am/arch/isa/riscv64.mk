MARCH ?= rv64gc

ifeq ($(TOOLCHAIN), LLVM)
CROSS_COMPILE :=
else ifeq ($(LINUX_GNU_TOOLCHAIN),1)
CROSS_COMPILE := riscv64-linux-gnu-
else
CROSS_COMPILE := riscv64-unknown-linux-gnu-
endif

# clang requires an explicit --target; a clang built with Target:unknown will
# reject any arch-specific flag (including -mcmodel=medany) without it.
ifeq ($(TOOLCHAIN), LLVM)
  LLVM_TARGET  ?= --target=riscv64-unknown-linux-gnu
  MCMODEL      := -mcmodel=medany
endif

COMMON_FLAGS  := -fno-pic $(LLVM_TARGET) -march=$(MARCH) $(MCMODEL)

CFLAGS        += $(COMMON_FLAGS) -static
ASFLAGS       += $(COMMON_FLAGS) -O0
LDFLAGS       += -melf64lriscv
