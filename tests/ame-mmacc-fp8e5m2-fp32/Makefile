NAME = ame-mmacc-fp8e5m2-fp32
SRCS = mat.c matop.S

# Zames proposal-14 matrix extension.
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync

include $(AM_HOME)/Makefile.app
