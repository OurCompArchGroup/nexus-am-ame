NAME = ame-ls_word
SRCS =  mat.c matop.S

# llvm-mc -triple=riscv64 -mattr=+zame,+matrix-xuantie-0.6 ../matop.S -filetype=obj 
LLVM_MC = $(LLVM_HOME)/bin/llvm-mc

APP_DIR ?= $(shell pwd)
INC_DIR += $(APP_DIR)/include/
DST_DIR ?= $(APP_DIR)/build/$(ARCH)/
define llvm-mc-rule
$(DST_DIR)/$(notdir $(1:.S=.o)): $(1)
	@echo "[LLVM-MC] Assembling $$< with Matrix extensions"
	@mkdir -p $(DST_DIR)
	$(LLVM_MC) -triple=riscv64 -mattr=+zame,+matrix-xuantie-0.6 -filetype=obj $$< -o $$@
endef



$(eval $(call llvm-mc-rule,matop.S))

include $(AM_HOME)/Makefile.app
