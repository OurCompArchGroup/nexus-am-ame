# Measured OpenLLC GEMM source

This frozen, always-checked source produced the fourteen 512×512×1024 OpenLLC measurements. It uses the normal nexus-am app Makefile and supports MODE, SCHEDULE, M, N, K, SEED and OUTPUT_ROOT. Build from the xsai-env root after `source env.sh`:

```sh
make -C nexus-am/apps/ame-bitmte-ppa/reproduce/measured-v8 image \
  MODE=i8-fp2pack4 SCHEDULE=npair M=512 N=512 K=1024 SEED=0x20261103
```

The app root separately supports CHECK=0 by default for power workloads and explicit CHECK=1. This snapshot always checks every output after the measured ROI. Use this snapshot to reproduce the measured image layout and cycles. The complete OpenLLC workflow is in xsai-env `docs/bmte/openllc-performance/README.md`.
