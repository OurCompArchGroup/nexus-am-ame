NAME = ame-gemm-fuzz
SRCS = test.c ame.c

override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH
export TOOLCHAIN

# Proposal-14/Zames randomized GEMM test with a scalar reference oracle.
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync
CFLAGS += -Wno-unused-function -Wno-unused-variable

include $(AM_HOME)/Makefile.app
