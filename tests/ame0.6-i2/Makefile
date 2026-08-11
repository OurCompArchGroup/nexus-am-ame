NAME = ametest-i2
SRCS = test.c ame.c

override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH
export TOOLCHAIN

MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync
CFLAGS += -Wno-unused-function -Wno-unused-variable

include $(AM_HOME)/Makefile.app

run-nemu: default

run-emu: default
