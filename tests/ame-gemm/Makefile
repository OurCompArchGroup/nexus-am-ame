NAME = ame-gemm
SRCS =  mat.c matop.S


# Zames matrix extension: assembled directly by clang via -march extensions.
# zames     = main-body matrix instructions
# zmasync   = optional async store synchronization extension
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync

include $(AM_HOME)/Makefile.app
