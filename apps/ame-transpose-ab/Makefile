SRCS = mat.c matop.S

# Select LLVM before Makefile.app includes the RISC-V architecture flags.
# This enables clang's target triple and medany code model for the 0x80000000
# AM image layout, in addition to selecting the in-tree compiler binaries.
# Export it so recursive AM/klib builds use the same target and code model.
TOOLCHAIN ?= LLVM
export TOOLCHAIN

TRANSPOSE_PROBE_MASK ?= 1
ifeq ($(TRANSPOSE_PROBE_MASK),1)
PROBE_PRECISION := e8
else ifeq ($(TRANSPOSE_PROBE_MASK),2)
PROBE_PRECISION := e16
else ifeq ($(TRANSPOSE_PROBE_MASK),4)
PROBE_PRECISION := e32
else ifeq ($(TRANSPOSE_PROBE_MASK),8)
PROBE_PRECISION := e4
else
$(error TRANSPOSE_PROBE_MASK must be 1, 2, 4, or 8)
endif

NAME = ame-transpose-ab-macc-$(PROBE_PRECISION)
CPPFLAGS += -DTRANSPOSE_PROBE_MASK=$(TRANSPOSE_PROBE_MASK)
ASFLAGS += -DTRANSPOSE_PROBE_MASK=$(TRANSPOSE_PROBE_MASK)

# The precision selector changes preprocessed C and assembly, so each mode
# needs a distinct object directory to prevent stale cross-mode relinks.
DST_DIR = $(APP_DIR)/build/obj-$(ARCH)-probe-$(PROBE_PRECISION)/

# Assemble Zames matrix instructions with the in-tree LLVM toolchain.
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync

include $(AM_HOME)/Makefile.app
