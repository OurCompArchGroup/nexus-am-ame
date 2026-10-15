NAME = ame-bitmte
.DEFAULT_GOAL := image
override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH TOOLCHAIN

# Four independently runnable examples share the packing, kernel and checker.
EXAMPLE ?= i8-fp2pack4
EXAMPLES := i8-fp2pack4 i4-fp2pack4 i8-i1pack8 i4-i1pack8
ifeq ($(filter $(EXAMPLE),$(EXAMPLES)),)
$(error EXAMPLE must be one of: $(EXAMPLES))
endif
A_BITS := $(if $(filter i4-%,$(EXAMPLE)),4,8)
B_BITS := $(if $(filter %-fp2pack4,$(EXAMPLE)),2,1)
MODE_i8-fp2pack4 := BMTE_I8_TERNARY
MODE_i4-fp2pack4 := BMTE_I4_TERNARY
MODE_i8-i1pack8 := BMTE_I8_BINARY
MODE_i4-i1pack8 := BMTE_I4_BINARY
M ?= 256
N ?= 256
K ?= $(if $(filter 2,$(B_BITS)),512,1024)

# Shape-specific objects prevent stale binaries when switching M/N/K or mode.
OUTPUT_DIR := $(CURDIR)/build/$(EXAMPLE)/$(M)x$(N)x$(K)
DST_DIR := $(OUTPUT_DIR)/obj
BINARY := $(OUTPUT_DIR)/gemm
SRCS := main.c bitmte.c hw.c
CPPFLAGS += -DBITMTE_MODE=$(MODE_$(EXAMPLE)) -DBITMTE_A_BITS=$(A_BITS) \
  -DBITMTE_B_BITS=$(B_BITS) -DBITMTE_NAME='"$(EXAMPLE)"' \
  -DBITMTE_M=$(M) -DBITMTE_N=$(N) -DBITMTE_K=$(K)
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync_xbitmte
OBJDUMP_FLAGS = --mattr=+zames,+zmasync,+xbitmte

.PHONY: help
help:
	@echo 'make image EXAMPLE=<$(EXAMPLES)> M=256 N=256 K=<512 ternary | 1024 binary>'
	@echo 'M,N: positive multiples of 128; K: positive multiple of 256 (ternary) or 512 (binary).'
	@echo 'Output: build/<EXAMPLE>/<M>x<N>x<K>/gemm.{elf,bin,txt}'
	@echo 'Run from xsai-env root: make run-nemu PAYLOAD=<absolute path to gemm.bin>'

ifneq ($(MAKECMDGOALS),help)
include $(AM_HOME)/Makefile.app
endif
