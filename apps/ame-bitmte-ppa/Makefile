NAME = ame-bitmte-ppa
.DEFAULT_GOAL := image
override ARCH := riscv64-xs
override TOOLCHAIN := LLVM
export ARCH TOOLCHAIN

MODE ?= i8-fp2pack4
MODES := i8-i8 bf16-bf16 i8-fp2pack4 i4-fp2pack4 i8-i1pack8 i4-i1pack8
ifeq ($(filter $(MODE),$(MODES)),)
$(error MODE must be one of: $(MODES))
endif
MODE_i8-i8 := 0
MODE_bf16-bf16 := 1
MODE_i8-fp2pack4 := 2
MODE_i4-fp2pack4 := 3
MODE_i8-i1pack8 := 4
MODE_i4-i1pack8 := 5
SCHEDULE ?= npair
ifeq ($(SCHEDULE),npair)
SCHEDULE_CODE := 0
else ifeq ($(SCHEDULE),mpair)
SCHEDULE_CODE := 1
else ifeq ($(SCHEDULE),npair-fixed)
SCHEDULE_CODE := 2
else ifeq ($(SCHEDULE),quad-roll)
SCHEDULE_CODE := 3
else ifeq ($(SCHEDULE),mpair-bdefer)
ifeq ($(filter $(MODE),i8-i8 bf16-bf16),)
$(error mpair-bdefer is only for i8-i8 or bf16-bf16)
endif
SCHEDULE_CODE := 4
else
$(error SCHEDULE must be npair, mpair, npair-fixed, quad-roll or mpair-bdefer)
endif
M ?= 256
N ?= 256
K ?= 512
SEED ?= 0x20261103
CHECK ?= 0
ifneq ($(CHECK),0)
ifneq ($(CHECK),1)
$(error CHECK must be 0 (skip result verification) or 1 (verify all results))
endif
endif
ifeq ($(SCHEDULE),mpair-bdefer)
ifneq ($(filter 256x512x512 256x512x1024 512x512x512 512x512x1024,$(M)x$(N)x$(K)),$(M)x$(N)x$(K))
$(error mpair-bdefer is frozen to MxNxK=256x512x512|256x512x1024|512x512x512|512x512x1024)
endif
ifneq ($(SEED),0x20261103)
$(error mpair-bdefer is frozen to SEED=0x20261103)
endif
endif
OUTPUT_ROOT ?= $(CURDIR)/build
OUTPUT_DIR := $(OUTPUT_ROOT)/$(MODE)/$(SCHEDULE)/$(M)x$(N)x$(K)-s$(SEED)-check$(CHECK)
DST_DIR := $(OUTPUT_DIR)/obj
BINARY := $(OUTPUT_DIR)/gemm
SRCS := main.c
ORACLE_HEADER := $(OUTPUT_DIR)/expected.h
CPPFLAGS += -DPPA_MODE=$(MODE_$(MODE)) -DPPA_MODE_NAME='"$(MODE)"' \
  -DPPA_SCHEDULE=$(SCHEDULE_CODE) -DPPA_SCHEDULE_NAME='"$(SCHEDULE)"' \
  -DPPA_M=$(M) -DPPA_N=$(N) -DPPA_K=$(K) -DPPA_SEED=$(SEED) -DPPA_CHECK=$(CHECK) -I$(OUTPUT_DIR)
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync_xbitmte
OBJDUMP_FLAGS = --mattr=+zames,+zmasync,+xbitmte

.PHONY: help
help:
	@echo 'make image MODE=<$(MODES)> SCHEDULE=<npair|mpair|npair-fixed|quad-roll|mpair-bdefer> M=256 N=256 K=512 SEED=0x20261103 CHECK=<0|1>'
	@echo 'CHECK=0 (default): skip post-kernel result verification; CHECK=1: verify every output.'
	@echo 'M/N: positive multiples of128; K: positive multiple of512 for the common six-mode experiment.'
	@echo 'Output: $(OUTPUT_DIR)/gemm.{elf,bin,txt}'

ifneq ($(MAKECMDGOALS),help)
$(ORACLE_HEADER): $(CURDIR)/oracle.py
	python3 $< --mode $(MODE) --m $(M) --n $(N) --k $(K) --seed $(SEED) --out $@

$(DST_DIR)/main.o: $(ORACLE_HEADER)

include $(AM_HOME)/Makefile.app
endif
