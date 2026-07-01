NAME = ame-gemm-blas
SRCS = $(wildcard *.c)

# This app links the xsai-blas archive and calls its public GEMM API instead of
# shipping a hand-written matop.S. The archive is built (via CMake) with the
# library's own AME march flags, so this app stays plain C: no llvm-mc rule and
# no matrix-extension flags here.
#
# xsai-blas/Makefile (app-mode branch, triggered because NAME is set) supplies:
#   - LINK_FILES += $(XSAI_BLAS_ARCHIVE)
#   - INC_DIR for include/, kernel/, platform/include/
#   - a rule that builds the archive with KERNEL/RUNTIME before `image`
#   - include $(AM_HOME)/Makefile.app   (so we must NOT include it again here)
# Locate xsai-blas. The canonical, layout-independent way is the XSAI_BLAS_HOME
# env var (xsai-blas/init.sh exports it: `cd xsai-blas && source init.sh`); that
# always wins here via ?=.

ifndef XSAI_BLAS_HOME
  $(error XSAI_BLAS_HOME is not set — run: cd xsai-blas && source init.sh)
endif
KERNEL  ?= ame
RUNTIME ?= am

include $(XSAI_BLAS_HOME)/Makefile
