NAME = ame-zero
SRCS =  mat.c matop.S

# llvm-mc -triple=riscv64 -mattr=+zame,+matrix-xuantie-0.6 ../matop.S -filetype=obj 
LLVM_MC = /nfs/home/zhuhonghua/xsai/xsai-env/local/llvm/bin/llvm-mc
define llvm-mc-rule
$(1:.S=.o): $(1)
	@echo "[LLVM-MC] Assembling $$< with Matrix extensions"
	$(LLVM_MC) -triple=riscv64 -mattr=+zame,+matrix-xuantie-0.6 -filetype=obj $$< -o $$@
	mv $$@ build/$(ARCH)/
endef

$(eval $(call llvm-mc-rule,matop.S))
include $(AM_HOME)/Makefile.app
