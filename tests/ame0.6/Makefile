NAME = ametest
SRCS = test.c ame.c

override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH
export TOOLCHAIN

# Legacy directory name retained for continuity; the test body uses AME
# proposal-12/Zames instructions.
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync
CFLAGS += -Wno-unused-function -Wno-unused-variable

include $(AM_HOME)/Makefile.app
