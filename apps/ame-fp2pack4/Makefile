NAME = ame-fp2pack4
SRCS = main.c

override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH
export TOOLCHAIN

MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync

include $(AM_HOME)/Makefile.app
