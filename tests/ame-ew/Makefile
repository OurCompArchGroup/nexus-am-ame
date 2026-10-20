NAME = ame-ew
SRCS = main.c ew.c reduce.c memory.c convert.c operators.c

ARCH ?= riscv64-xs
TOOLCHAIN ?= LLVM
MARCH = rv64gc_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync
# Keep scalar reference arithmetic separate; no RVV or host libm dependency.
CFLAGS += -ffp-contract=off

include $(AM_HOME)/Makefile.app
