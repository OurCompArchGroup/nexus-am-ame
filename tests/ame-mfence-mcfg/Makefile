NAME = ame-mfence-mcfg
SRCS = mat.c matop.S

# Zames matrix extension assembled directly by clang via -march extensions.
#   zames    = main matrix instructions (msetcfg/mgetcfg, mla/mlb/mlc, msc, mmacc)
#   zmasync  = synchronization extension providing mfence / msyncregreset / mrelease / macquire
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync

include $(AM_HOME)/Makefile.app
