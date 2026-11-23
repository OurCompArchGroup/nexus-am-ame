NAME = ame-gemm
SRCS =  mat.c matop.S


# Zames matrix extension: assembled directly by clang via -march extensions.
# zames     = main-body matrix instructions
# zmasync   = optional async store synchronization extension
MARCH = rv64gc_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt_zames_zmasync
OBJDUMP_FLAGS ?= --mattr=+zames,+zmasync

MXFP_CPU_QUANT ?= 0
MXFP_CONVERT_LOAD ?= 0
MXFP_QUANT_ONLY ?= 0
MXFP_MODEL_ACTIVATION ?= 0
MXFP_MODEL_WEIGHT_OBJ ?=
MXFP_MODEL_WEIGHT_FILE ?=
MXFP_MODEL_LMHEAD ?= 0
MXFP_MODEL_LMHEAD_OUTER ?= 0
MXFP_MODEL_LMHEAD_WEIGHT_FILE ?=
MXFP_MODEL_LMHEAD_REFERENCE_FILE ?=
MXFP_MODEL_N1_CASE ?= stories-head
MXFP_MODEL_DOT_K288 ?= 0
MXFP_MODEL_DOT_K1024 ?= 0
MXFP_MODEL_SET ?= stories
MXFP_MODEL_RVV_AMAX ?= 0
MXFP_MODEL_PACKED_PHASE ?= 0
MXFP_MODEL_PREP_SPLIT ?= 0
F16_MODEL_ATTENTION ?= 0
F16_MODEL_ATTENTION_SPLIT ?= 0
F16_MODEL_ATTENTION_SOFTMAX ?= 0
F16_MODEL_SOFTMAX_LOGITS_FILE ?=
F16_MODEL_ATTENTION_ROPE ?= 0
MXFP_RMS_ONLY ?= 0
MXFP_RMS_DYNAMIC ?= 0
# External operator attestation of the frozen config/model, NOT CSR discovery.
MXFP_RMS_DYNAMIC_ATTEST ?= none
MXFP_PERF_MN ?= 512
ifeq ($(filter $(MXFP_PERF_MN),128 512),)
$(error MXFP_PERF_MN must be 128 or 512; performance K remains 512)
endif
ifeq ($(MXFP_CONVERT_LOAD),1)
ifneq ($(MXFP_CPU_QUANT),1)
$(error MXFP_CONVERT_LOAD=1 requires MXFP_CPU_QUANT=1 for real preflight/fallback)
endif
endif
ifeq ($(MXFP_QUANT_ONLY),1)
ifneq ($(MXFP_CPU_QUANT),1)
$(error MXFP_QUANT_ONLY=1 requires MXFP_CPU_QUANT=1)
endif
ifneq ($(MXFP_CONVERT_LOAD),0)
$(error MXFP_QUANT_ONLY=1 excludes MXFP_CONVERT_LOAD modes)
endif
endif
ifeq ($(MXFP_RMS_ONLY),1)
ifneq ($(MXFP_CPU_QUANT):$(MXFP_QUANT_ONLY),1:1)
$(error MXFP_RMS_ONLY=1 requires MXFP_CPU_QUANT=1 MXFP_QUANT_ONLY=1)
endif
endif
ifeq ($(MXFP_RMS_DYNAMIC),1)
ifneq ($(MXFP_CPU_QUANT):$(MXFP_QUANT_ONLY):$(MXFP_RMS_ONLY),1:1:1)
$(error MXFP_RMS_DYNAMIC=1 requires MXFP_CPU_QUANT=1 MXFP_QUANT_ONLY=1 MXFP_RMS_ONLY=1)
endif
endif
ifeq ($(MXFP_MODEL_ACTIVATION),1)
ifneq ($(MXFP_CPU_QUANT):$(MXFP_QUANT_ONLY):$(MXFP_RMS_ONLY),1:1:0)
$(error MXFP_MODEL_ACTIVATION=1 requires MXFP_CPU_QUANT=1 MXFP_QUANT_ONLY=1 without MXFP_RMS_ONLY)
endif
ifneq ($(MXFP_MODEL_SET),stories)
ifneq ($(MXFP_MODEL_SET),qwen06)
$(error MXFP_MODEL_SET must be stories or qwen06)
endif
endif
ifneq ($(MXFP_MODEL_RVV_AMAX),0)
ifneq ($(MXFP_MODEL_RVV_AMAX),1)
$(error MXFP_MODEL_RVV_AMAX must be 0 or 1)
endif
endif
ifneq ($(MXFP_MODEL_PACKED_PHASE),0)
ifneq ($(MXFP_MODEL_PACKED_PHASE),1)
$(error MXFP_MODEL_PACKED_PHASE must be 0 or 1)
endif
endif
ifneq ($(MXFP_MODEL_PREP_SPLIT),0)
ifneq ($(MXFP_MODEL_PREP_SPLIT),1)
$(error MXFP_MODEL_PREP_SPLIT must be 0 or 1)
endif
ifneq ($(MXFP_MODEL_PACKED_PHASE),0)
$(error MXFP_MODEL_PREP_SPLIT=1 requires MXFP_MODEL_PACKED_PHASE=0)
endif
endif
ifeq ($(wildcard $(MXFP_PACK_ASM)),)
$(error MXFP_MODEL_ACTIVATION=1 requires the sealed PACK source in MXFP_PACK_ASM)
endif
endif
ifeq ($(F16_MODEL_ATTENTION),1)
ifneq ($(MXFP_CPU_QUANT):$(MXFP_QUANT_ONLY):$(MXFP_MODEL_ACTIVATION):$(MXFP_RMS_ONLY),1:1:0:0)
$(error F16_MODEL_ATTENTION=1 requires MXFP_CPU_QUANT=1 MXFP_QUANT_ONLY=1 without other model modes)
endif
ifeq ($(filter $(MXFP_MODEL_SET),stories qwen06),)
$(error F16 attention requires MXFP_MODEL_SET=stories or qwen06)
endif
ifeq ($(MXFP_MODEL_SET):$(F16_MODEL_ATTENTION_ROPE),qwen06:1)
$(error Qwen F16 attention does not cover RoPE)
endif
endif
ifneq ($(filter 1,$(F16_MODEL_ATTENTION_SPLIT) $(F16_MODEL_ATTENTION_SOFTMAX) $(F16_MODEL_ATTENTION_ROPE)),)
ifneq ($(F16_MODEL_ATTENTION),1)
$(error F16 attention probes require F16_MODEL_ATTENTION=1)
endif
endif
ifneq ($(strip $(F16_MODEL_SOFTMAX_LOGITS_FILE)),)
ifneq ($(MXFP_MODEL_SET):$(F16_MODEL_ATTENTION_SOFTMAX),stories:1)
$(error F16_MODEL_SOFTMAX_LOGITS_FILE requires the Stories Softmax fixture)
endif
ifeq ($(wildcard $(F16_MODEL_SOFTMAX_LOGITS_FILE)),)
$(error F16_MODEL_SOFTMAX_LOGITS_FILE must exist)
endif
endif
ifneq ($(strip $(MXFP_MODEL_WEIGHT_OBJ)),)
ifneq ($(MXFP_CPU_QUANT):$(MXFP_QUANT_ONLY):$(MXFP_MODEL_ACTIVATION):$(MXFP_RMS_ONLY):$(F16_MODEL_ATTENTION),1:1:0:0:0)
$(error MXFP_MODEL_WEIGHT_OBJ requires MXFP_CPU_QUANT=1 MXFP_QUANT_ONLY=1 without other model modes)
endif
ifeq ($(wildcard $(MXFP_MODEL_WEIGHT_OBJ)),)
$(error MXFP_MODEL_WEIGHT_OBJ must be the pruned, matched AME weight owner object)
endif
endif
ifneq ($(strip $(MXFP_MODEL_WEIGHT_FILE)),)
ifeq ($(strip $(MXFP_MODEL_WEIGHT_OBJ)),)
$(error MXFP_MODEL_WEIGHT_FILE requires MXFP_MODEL_WEIGHT_OBJ)
endif
ifeq ($(wildcard $(MXFP_MODEL_WEIGHT_FILE)),)
$(error MXFP_MODEL_WEIGHT_FILE must exist)
endif
endif
ifneq ($(MXFP_MODEL_N1_CASE),stories-head)
ifneq ($(MXFP_MODEL_LMHEAD):$(MXFP_MODEL_DOT_K288),1:0)
$(error Qwen N1 cases require MXFP_MODEL_LMHEAD=1 MXFP_MODEL_DOT_K288=0)
endif
endif
ifeq ($(MXFP_MODEL_LMHEAD_OUTER),1)
ifneq ($(MXFP_MODEL_LMHEAD),1)
$(error MXFP_MODEL_LMHEAD_OUTER=1 requires MXFP_MODEL_LMHEAD=1)
endif
ifeq ($(filter $(MXFP_MODEL_N1_CASE),stories-head qwen-head qwen-ffn-expand),)
$(error MXFP_MODEL_LMHEAD_OUTER=1 requires a supported N1 projection)
endif
endif
ifneq ($(strip $(MXFP_MODEL_LMHEAD_WEIGHT_FILE)),)
ifneq ($(MXFP_MODEL_LMHEAD_OUTER),1)
$(error MXFP_MODEL_LMHEAD_WEIGHT_FILE requires MXFP_MODEL_LMHEAD_OUTER=1)
endif
ifeq ($(wildcard $(MXFP_MODEL_LMHEAD_WEIGHT_FILE)),)
$(error MXFP_MODEL_LMHEAD_WEIGHT_FILE must exist)
endif
endif
ifneq ($(strip $(MXFP_MODEL_LMHEAD_REFERENCE_FILE)),)
ifeq ($(strip $(MXFP_MODEL_LMHEAD_WEIGHT_FILE)),)
$(error MXFP_MODEL_LMHEAD_REFERENCE_FILE requires MXFP_MODEL_LMHEAD_WEIGHT_FILE)
endif
ifeq ($(filter $(MXFP_MODEL_N1_CASE),qwen-head qwen-ffn-expand),)
$(error MXFP_MODEL_LMHEAD_REFERENCE_FILE requires a Qwen K1024 N1 case)
endif
ifeq ($(wildcard $(MXFP_MODEL_LMHEAD_REFERENCE_FILE)),)
$(error MXFP_MODEL_LMHEAD_REFERENCE_FILE must exist)
endif
endif
ifeq ($(MXFP_MODEL_LMHEAD),1)
ifneq ($(MXFP_CPU_QUANT):$(MXFP_QUANT_ONLY):$(MXFP_MODEL_ACTIVATION):$(MXFP_RMS_ONLY):$(F16_MODEL_ATTENTION):$(strip $(MXFP_MODEL_WEIGHT_OBJ)),1:1:0:0:0:)
$(error MXFP_MODEL_LMHEAD=1 requires MXFP_CPU_QUANT=1 MXFP_QUANT_ONLY=1 without other model modes)
endif
ifeq ($(filter $(MXFP_MODEL_DOT_K288),0 1),)
$(error MXFP_MODEL_DOT_K288 must be 0 or 1)
endif
ifeq ($(filter $(MXFP_MODEL_DOT_K1024),0 1),)
$(error MXFP_MODEL_DOT_K1024 must be 0 or 1)
endif
ifeq ($(MXFP_MODEL_DOT_K1024),1)
ifeq ($(filter $(MXFP_MODEL_N1_CASE),qwen-head qwen-ffn-expand),)
$(error MXFP_MODEL_DOT_K1024=1 requires a Qwen K1024 N1 case)
endif
endif
ifeq ($(MXFP_MODEL_N1_CASE),stories-head)
MXFP_MODEL_N1_ROWS := 32000
MXFP_MODEL_N1_BLOCKS := 9
else ifeq ($(MXFP_MODEL_N1_CASE),qwen-head)
MXFP_MODEL_N1_ROWS := 151936
MXFP_MODEL_N1_BLOCKS := 32
else ifeq ($(MXFP_MODEL_N1_CASE),qwen-ffn-expand)
MXFP_MODEL_N1_ROWS := 3072
MXFP_MODEL_N1_BLOCKS := 32
else ifeq ($(MXFP_MODEL_N1_CASE),qwen-ffn-contract)
MXFP_MODEL_N1_ROWS := 1024
MXFP_MODEL_N1_BLOCKS := 96
else
$(error MXFP_MODEL_N1_CASE must be stories-head, qwen-head, qwen-ffn-expand, or qwen-ffn-contract)
endif
endif
ifeq ($(MXFP_CPU_QUANT),1)
override MARCH := rv64gcv_zvfh_zba_zbb_zames_zmasync
MXFP_QUANT_BUILD := $(CURDIR)/build/r24-cpuquant
ifeq ($(MXFP_QUANT_ONLY),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/r33-quant-only
endif
ifeq ($(MXFP_CONVERT_LOAD),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/r31-convert
endif
ifeq ($(MXFP_PERF_MN),128)
MXFP_QUANT_BUILD := $(MXFP_QUANT_BUILD)-128x128x512
endif
ifeq ($(MXFP_RMS_ONLY),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/r38-rms512-control
endif
ifeq ($(MXFP_RMS_DYNAMIC),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/rmsdyn512-am
endif
MXFP_QUANT_SYSROOT ?= $(RISCV_SYSROOT)
MXFP_GGML := $(abspath $(AM_HOME)/../software/llama.cpp/ggml)
MXFP_GGML_SRCS := $(MXFP_GGML)/src/ggml-cpu/quants.c \
                  $(MXFP_GGML)/src/ggml-quants.c \
                  $(MXFP_GGML)/src/ggml-cpu/arch/riscv/ame/ame-helper.c
MODEL_GGML_ENTRY_SRCS := $(MXFP_GGML)/src/ggml-cpu/ggml-cpu.c $(MXFP_GGML)/src/ggml-cpu/llamafile/sgemm.cpp
ifeq ($(MXFP_MODEL_ACTIVATION),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/mxfp-model-$(MXFP_MODEL_SET)-amax$(MXFP_MODEL_RVV_AMAX)$(if $(filter 1,$(MXFP_MODEL_PACKED_PHASE)),-phase)$(if $(filter 1,$(MXFP_MODEL_PREP_SPLIT)),-layout)
MXFP_GGML_SRCS += $(MXFP_GGML)/src/ggml-cpu/arch/riscv/ame/ame-matmul.c
SRCS := $(filter-out matop.S,$(SRCS)) $(MXFP_PACK_ASM)
endif
ifeq ($(F16_MODEL_ATTENTION),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/f16-model-attention$(if $(filter qwen06,$(MXFP_MODEL_SET)),-qwen06)$(if $(filter 1,$(F16_MODEL_ATTENTION_SPLIT)),-split)
MXFP_GGML_SRCS += $(MODEL_GGML_ENTRY_SRCS)
ifneq ($(filter 1,$(F16_MODEL_ATTENTION_SOFTMAX) $(F16_MODEL_ATTENTION_ROPE)),)
MXFP_GGML_SRCS += $(MXFP_GGML)/src/ggml-cpu/ops.cpp
endif
ifeq ($(F16_MODEL_ATTENTION_SOFTMAX),1)
MXFP_QUANT_BUILD := $(MXFP_QUANT_BUILD)-softmax
MXFP_GGML_SRCS += $(MXFP_GGML)/src/ggml-cpu/vec.cpp
endif
ifeq ($(F16_MODEL_ATTENTION_ROPE),1)
MXFP_QUANT_BUILD := $(MXFP_QUANT_BUILD)-rope
MXFP_GGML_SRCS += $(MXFP_GGML)/src/ggml.c
endif
endif
ifneq ($(strip $(MXFP_MODEL_WEIGHT_OBJ)),)
MXFP_QUANT_BUILD := $(CURDIR)/build/mxfp-model-weight-$(notdir $(basename $(MXFP_MODEL_WEIGHT_OBJ)))
ifneq ($(strip $(MXFP_MODEL_WEIGHT_FILE)),)
MXFP_QUANT_BUILD := $(MXFP_QUANT_BUILD)-file
endif
endif
ifeq ($(MXFP_MODEL_LMHEAD),1)
MXFP_QUANT_BUILD := $(CURDIR)/build/mxfp-model-lmhead-$(MXFP_MODEL_DOT_K288)$(if $(filter-out stories-head,$(MXFP_MODEL_N1_CASE)),-$(MXFP_MODEL_N1_CASE))$(if $(filter 1,$(MXFP_MODEL_DOT_K1024)),-k1024)
MXFP_GGML_SRCS += $(MXFP_GGML)/src/ggml-cpu/arch/riscv/quants.c
ifeq ($(MXFP_MODEL_LMHEAD_OUTER),1)
MXFP_QUANT_BUILD := $(MXFP_QUANT_BUILD)-outer
MXFP_GGML_SRCS += $(MODEL_GGML_ENTRY_SRCS) $(MXFP_GGML)/src/ggml.c $(MXFP_GGML)/src/ggml-cpu/vec.cpp
endif
endif
SRCS += $(MXFP_GGML_SRCS)
# Never reuse app or runtime objects built with the prepared-only flags.
override DST_DIR := $(MXFP_QUANT_BUILD)/app
override BINARY := $(MXFP_QUANT_BUILD)/ame-gemm-$(ARCH)
endif

include $(AM_HOME)/Makefile.app

$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_PERF_MN=$(MXFP_PERF_MN)

ifeq ($(MXFP_CPU_QUANT),1)
ifneq ($(TOOLCHAIN),LLVM)
$(error MXFP_CPU_QUANT=1 requires the canonical LLVM AME toolchain)
endif
MXFP_GGML_OBJS := $(addprefix $(DST_DIR)/,$(addsuffix .o,$(basename $(MXFP_GGML_SRCS))))
MXFP_GGML_PRUNED := $(MXFP_QUANT_BUILD)/mxfp4-source-pruned.o
MXFP_AM_ARCHIVE := $(MXFP_QUANT_BUILD)/am/am-$(ARCH).a
MXFP_KLIB_ARCHIVE := $(MXFP_QUANT_BUILD)/klib/klib-$(ARCH).a

# Use the existing Makefile.compile rule. Only the three canonical GGML TUs
# need section splitting and the GGML warning override after AM's -Werror.
$(MXFP_GGML_OBJS): private CFLAGS += -ffunction-sections -fdata-sections \
    --sysroot=$(MXFP_QUANT_SYSROOT) \
    -I$(MXFP_GGML)/include -I$(MXFP_GGML)/src -I$(MXFP_GGML)/src/ggml-cpu \
    -I$(MXFP_GGML)/src/ggml-cpu/arch/riscv/ame -Wno-unused-function
# The old modes keep their klib/common-block include path. RMS-only includes
# real GGML headers before AM printf macros, without klib's time declaration.
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_CPU_QUANT=1 \
    -I$(MXFP_GGML)/src -isystem $(MXFP_QUANT_SYSROOT)/usr/include
ifeq ($(MXFP_RMS_ONLY),1)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_RMS_ONLY=1 \
    -I$(MXFP_GGML)/include -I$(MXFP_GGML)/src/ggml-cpu
$(DST_DIR)/mat.o: private CFLAGS += -Wno-unused-function -ffp-contract=off
endif
ifeq ($(MXFP_CONVERT_LOAD),1)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_CONVERT_LOAD=1
# The canonical .S rule consumes ASFLAGS, not CPPFLAGS.
$(DST_DIR)/matop.o: private ASFLAGS += -DMXFP_CONVERT_LOAD=1
endif
ifeq ($(MXFP_RMS_DYNAMIC),1)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_RMS_DYNAMIC=1 \
    -DMXFP_RMS_DYNAMIC_ATTEST=\"$(MXFP_RMS_DYNAMIC_ATTEST)\"
$(DST_DIR)/matop.o: private ASFLAGS += -DMXFP_RMS_DYNAMIC=1
endif
ifeq ($(MXFP_QUANT_ONLY),1)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_QUANT_ONLY=1
$(DST_DIR)/mat.o: private CFLAGS += -ffunction-sections -fdata-sections
endif
ifneq ($(strip $(MXFP_MODEL_WEIGHT_OBJ)),)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_MODEL_WEIGHT=1 -I$(MXFP_GGML)/include
$(DST_DIR)/mat.o: private CFLAGS += -Wno-unused-function
ifneq ($(strip $(MXFP_MODEL_WEIGHT_FILE)),)
$(DST_DIR)/mat.o: $(abspath $(MXFP_MODEL_WEIGHT_FILE))
$(DST_DIR)/mat.o: private CPPFLAGS += \
    -DMXFP_MODEL_WEIGHT_FILE=\"$(abspath $(MXFP_MODEL_WEIGHT_FILE))\"
endif
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/arch/riscv/ame/ame-helper.o: private CPPFLAGS += \
    -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -DNDEBUG
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/arch/riscv/ame/ame-helper.o: private CFLAGS += \
    -O3 -fhosted -fbuiltin -march=rv64g_zfh_v_zvfh_zames_zmasync_zvl128b_zicbop_zihintpause
endif
ifeq ($(MXFP_MODEL_ACTIVATION),1)
MXFP_MODEL_DEFS := -DGGML_RV_MXFP4_PACKED -DGGML_AME_MXFP4_ADJACENT=1 \
    -DGGML_AME_MXFP4_REUSE_AMAX=1 -DGGML_MXFP4_SUPPLIED_BATCH8=1 \
    -DGGML_MXFP4_PACKED_PHASE_PROFILE=$(MXFP_MODEL_PACKED_PHASE) \
    -DGGML_AME_RMS_RVV_DOMAIN=1 -DGGML_AME_RMS_PREFLIGHT_BOUNDS=2 -DGGML_RMS_RVV_F64=1
$(DST_DIR)/mat.o: private CPPFLAGS += $(MXFP_MODEL_DEFS) -DMXFP_MODEL_ACTIVATION=1 \
    -I$(MXFP_GGML)/include
ifeq ($(MXFP_MODEL_SET),qwen06)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_MODEL_QWEN=1
endif
$(DST_DIR)/mat.o: private CFLAGS += -Wno-unused-function
$(MXFP_GGML_OBJS): private CPPFLAGS += $(MXFP_MODEL_DEFS) -D_GNU_SOURCE \
    -D_XOPEN_SOURCE=600 -DGGML_USE_RV_AME -DNDEBUG
$(MXFP_GGML_OBJS): private CFLAGS += -O3 -fhosted -fbuiltin \
    -march=rv64g_zfh_v_zvfh_zames_zmasync_zvl128b_zicbop_zihintpause
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/quants.o: private CPPFLAGS += \
    -DGGML_MXFP4_PREFLIGHT_RVV_AMAX=$(MXFP_MODEL_RVV_AMAX)
ifeq ($(MXFP_MODEL_PREP_SPLIT),1)
$(DST_DIR)/mat.o $(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/arch/riscv/ame/ame-matmul.o: private CPPFLAGS += \
    -DGGML_AME_MXFP4_PREP_SPLIT_PROFILE=1
MXFP_MODEL_PREP_SPLIT_ROOT := --undefined=ggml_ame_mx_prep_split_set
endif
$(DST_DIR)/$(basename $(MXFP_PACK_ASM)).o: private ASFLAGS += -DMXFP_RVV_PACK=1 \
    -DMXFP_RVV_PACK_BATCH=1 -DMXFP_RVV_PACK_PAIR=2 -DMXFP_RVV_PACK_LMUL=8
LDFLAGS += --wrap=ggml_abort
endif
ifeq ($(F16_MODEL_ATTENTION),1)
F16_ATTN_GGML_OBJS := $(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/ggml-cpu.o \
                     $(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/llamafile/sgemm.o
F16_ATTN_CXX_OBJS := $(filter %/sgemm.o,$(F16_ATTN_GGML_OBJS))
ifneq ($(filter 1,$(F16_MODEL_ATTENTION_SOFTMAX) $(F16_MODEL_ATTENTION_ROPE)),)
F16_ATTN_OPS := $(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/ops.o
F16_ATTN_GGML_OBJS += $(F16_ATTN_OPS)
F16_ATTN_CXX_OBJS += $(F16_ATTN_OPS)
endif
ifeq ($(F16_MODEL_ATTENTION_SOFTMAX),1)
F16_SOFTMAX_OBJS := $(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/vec.o
F16_ATTN_GGML_OBJS += $(F16_SOFTMAX_OBJS)
F16_ATTN_CXX_OBJS += $(F16_SOFTMAX_OBJS)
$(DST_DIR)/mat.o: private CPPFLAGS += -DF16_MODEL_ATTENTION_SOFTMAX=1
ifneq ($(strip $(F16_MODEL_SOFTMAX_LOGITS_FILE)),)
$(DST_DIR)/mat.o: $(abspath $(F16_MODEL_SOFTMAX_LOGITS_FILE))
$(DST_DIR)/mat.o: private CPPFLAGS += \
    -DF16_MODEL_SOFTMAX_LOGITS_FILE=\"$(abspath $(F16_MODEL_SOFTMAX_LOGITS_FILE))\"
endif
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/vec.o: private CPPFLAGS += -DGGML_AME_RVV_SOFTMAX_MASKED_EXP=1
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/vec.o: private CXXFLAGS += -Wno-error=uninitialized
endif
ifeq ($(F16_MODEL_ATTENTION_ROPE),1)
F16_ATTN_GGML_OBJS += $(DST_DIR)/$(MXFP_GGML)/src/ggml.o
$(DST_DIR)/$(MXFP_GGML)/src/ggml.o: private CPPFLAGS += -DGGML_VERSION=\"\" -DGGML_COMMIT=\"\"
$(DST_DIR)/mat.o: private CPPFLAGS += -DF16_MODEL_ATTENTION_ROPE=1
endif
$(DST_DIR)/mat.o: private CPPFLAGS += -DF16_MODEL_ATTENTION=1 \
    -DF16_MODEL_ATTENTION_QWEN=$(if $(filter qwen06,$(MXFP_MODEL_SET)),1,0) \
    -DF16_MODEL_ATTENTION_SPLIT=$(F16_MODEL_ATTENTION_SPLIT) \
    -I$(MXFP_GGML)/include -I$(MXFP_GGML)/src/ggml-cpu
$(DST_DIR)/mat.o: private CFLAGS += -Wno-unused-function -ffp-contract=off
$(F16_ATTN_GGML_OBJS): private CFLAGS += -O3 -fhosted -fbuiltin \
    -march=rv64g_zfh_v_zvfh_zames_zmasync_zvl128b_zicbop_zihintpause
$(F16_ATTN_GGML_OBJS): private CPPFLAGS += -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -DNDEBUG
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/ggml-cpu.o: private CPPFLAGS += \
    -DGGML_USE_LLAMAFILE -DGGML_USE_RV_AME -DGGML_XSAI_AM_F16_TEST=1
$(F16_ATTN_CXX_OBJS): private CXXFLAGS += \
    -std=c++17 -fhosted --gcc-toolchain=$(abspath $(MXFP_QUANT_SYSROOT)/..)
F16_ATTN_LINK_ROOTS := --undefined=llamafile_sgemm --undefined=ggml_cpu_fp32_to_fp16 \
    --undefined=ggml_xsai_am_single_thread_pool
ifeq ($(F16_MODEL_ATTENTION_SOFTMAX),1)
F16_ATTN_LINK_ROOTS += --undefined=ggml_compute_forward_soft_max
endif
LDFLAGS += --wrap=ggml_abort --wrap=ggml_log_internal
ifeq ($(F16_MODEL_ATTENTION_ROPE),1)
F16_ATTN_LINK_ROOTS += --undefined=ggml_compute_forward_rope $(filter --wrap=%,$(LDFLAGS))
endif
endif
ifeq ($(MXFP_MODEL_LMHEAD),1)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_MODEL_LMHEAD=1 -I$(MXFP_GGML)/include \
    -DMXFP_MODEL_N1_ROWS=$(MXFP_MODEL_N1_ROWS) -DMXFP_MODEL_N1_BLOCKS=$(MXFP_MODEL_N1_BLOCKS)
$(DST_DIR)/mat.o: private CFLAGS += -Wno-unused-function -ffp-contract=off
$(MXFP_GGML_OBJS): private CPPFLAGS += -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -DNDEBUG
$(MXFP_GGML_OBJS): private CFLAGS += -O3 -fhosted -fbuiltin \
    -march=rv64g_zfh_v_zvfh_zames_zmasync_zvl128b_zicbop_zihintpause
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/quants.o: private CPPFLAGS += \
    -DGGML_MXFP4_DOT_K288_RVV=$(MXFP_MODEL_DOT_K288) \
    -DGGML_MXFP4_DOT_K1024_RVV=$(MXFP_MODEL_DOT_K1024)
MXFP_MODEL_LMHEAD_ROOTS := --undefined=ggml_vec_dot_mxfp4_q8_0 --undefined=quantize_row_q8_0
ifeq ($(MXFP_MODEL_LMHEAD_OUTER),1)
$(DST_DIR)/mat.o: private CPPFLAGS += -DMXFP_MODEL_LMHEAD_OUTER=1
ifneq ($(strip $(MXFP_MODEL_LMHEAD_WEIGHT_FILE)),)
$(DST_DIR)/mat.o: $(abspath $(MXFP_MODEL_LMHEAD_WEIGHT_FILE))
$(DST_DIR)/mat.o: private CPPFLAGS += \
    -DMXFP_MODEL_LMHEAD_WEIGHT_FILE=\"$(abspath $(MXFP_MODEL_LMHEAD_WEIGHT_FILE))\"
endif
ifneq ($(strip $(MXFP_MODEL_LMHEAD_REFERENCE_FILE)),)
$(DST_DIR)/mat.o: $(abspath $(MXFP_MODEL_LMHEAD_REFERENCE_FILE))
$(DST_DIR)/mat.o: private CPPFLAGS += \
    -DMXFP_MODEL_LMHEAD_REFERENCE_FILE=\"$(abspath $(MXFP_MODEL_LMHEAD_REFERENCE_FILE))\"
endif
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/ggml-cpu.o: private CPPFLAGS += \
    -DGGML_USE_LLAMAFILE -DGGML_USE_RV_AME -DGGML_XSAI_AM_F16_TEST=1
$(DST_DIR)/$(MXFP_GGML)/src/ggml.o: private CPPFLAGS += -DGGML_VERSION=\"\" -DGGML_COMMIT=\"\"
$(filter %/sgemm.o %/vec.o,$(MXFP_GGML_OBJS)): private CXXFLAGS += \
    -std=c++17 -fhosted --gcc-toolchain=$(abspath $(MXFP_QUANT_SYSROOT)/..)
$(DST_DIR)/$(MXFP_GGML)/src/ggml-cpu/vec.o: private CXXFLAGS += -Wno-error=uninitialized
LDFLAGS += --wrap=ggml_abort --wrap=ggml_log_internal --wrap=printf --wrap=putchar
MXFP_MODEL_LMHEAD_ROOTS += --undefined=ggml_compute_forward_mul_mat \
    --undefined=ggml_xsai_am_single_thread_pool $(filter --wrap=%,$(LDFLAGS))
endif
endif

# R22's source-only GC seam removes unreachable quantizers before archive
# extraction; the final AM link still resolves real libm/libc and TLS errno.
$(MXFP_GGML_PRUNED): $(MXFP_GGML_OBJS) $(MXFP_MODEL_WEIGHT_OBJ)
	@$(LD) -r --gc-sections -melf64lriscv \
	  --undefined=quantize_row_mxfp4 \
	  --undefined=quantize_row_mxfp4_ref \
	  --undefined=ggml_ame_pack_mxfp4_blocks \
	  $(if $(filter 1,$(MXFP_RMS_DYNAMIC)),--undefined=dequantize_row_mxfp4) \
	  $(if $(filter 1,$(MXFP_QUANT_ONLY)),--undefined=ggml_mxfp4_prepare_supplied_scales) \
	  $(if $(filter 1,$(MXFP_CONVERT_LOAD)),--undefined=ggml_mxfp4_prepare_supplied_scales) \
	  $(if $(filter 1,$(MXFP_MODEL_ACTIVATION)),--undefined=ggml_ame_mx_prepare_activation \
	    --undefined=ggml_mxfp4_packed_stats_reset --undefined=ggml_mxfp4_packed_stats_get) \
	  $(if $(strip $(MXFP_MODEL_WEIGHT_OBJ)),--undefined=ggml_ame_test_weight_match_set \
	    --undefined=ggml_ame_test_weight_match) \
	  $(MXFP_MODEL_PREP_SPLIT_ROOT) $(F16_ATTN_LINK_ROOTS) $(MXFP_MODEL_LMHEAD_ROOTS) \
	  -o $@ $^

# Override only these app prerequisites, using the canonical library rules
# with explicit output paths so command-line app DST_DIR cannot leak in.
am:
	@$(MAKE) -s -C $(AM_HOME)/am AM_HOME="$(AM_HOME)" ARCH="$(ARCH)" \
	  TOOLCHAIN=LLVM LLVM_HOME="$(LLVM_HOME)" MARCH="$(MARCH)" \
	  CC_OPT="$(CC_OPT)" AR="$(LLVM_HOME)/bin/llvm-ar" \
	  DST_DIR="$(MXFP_QUANT_BUILD)/am" ARCHIVE="$(MXFP_AM_ARCHIVE)"

klib:
	@$(MAKE) -s -C $(AM_HOME)/libs/klib AM_HOME="$(AM_HOME)" ARCH="$(ARCH)" \
	  TOOLCHAIN=LLVM LLVM_HOME="$(LLVM_HOME)" MARCH="$(MARCH)" \
	  CC_OPT="$(CC_OPT)" AR="$(LLVM_HOME)/bin/llvm-ar" \
	  DST_DIR="$(MXFP_QUANT_BUILD)/klib" ARCHIVE="$(MXFP_KLIB_ARCHIVE)"

LINK_FILES = $(filter-out $(MXFP_GGML_OBJS),$(OBJS)) $(MXFP_GGML_PRUNED) \
             $(MXFP_AM_ARCHIVE) $(MXFP_KLIB_ARCHIVE) \
             $(MXFP_QUANT_SYSROOT)/usr/lib/libm.a \
             $(MXFP_QUANT_SYSROOT)/usr/lib/libc.a
image: $(MXFP_GGML_PRUNED)
endif
