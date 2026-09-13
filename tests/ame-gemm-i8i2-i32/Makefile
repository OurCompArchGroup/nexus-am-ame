M ?= 256
N ?= 256
K ?= 128
VERIFY_FULL ?= 0

# Each configuration owns its object files and linked payload.
NAME = ame-gemm-i8i2-i32-m$(M)-n$(N)-k$(K)-vf$(VERIFY_FULL)
SRCS = main.c
DST_DIR := $(CURDIR)/build/m$(M)-n$(N)-k$(K)-vf$(VERIFY_FULL)/riscv64-xs/
BINARY := $(CURDIR)/build/$(NAME)-riscv64-xs

override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH TOOLCHAIN

MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync
CFLAGS += -DBENCH_M=$(M) -DBENCH_N=$(N) -DBENCH_K=$(K) -DBENCH_VERIFY_FULL=$(VERIFY_FULL)

include $(AM_HOME)/Makefile.app

