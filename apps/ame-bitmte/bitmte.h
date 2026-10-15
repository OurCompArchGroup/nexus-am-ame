#ifndef AME_BITMTE_H
#define AME_BITMTE_H

#include <stddef.h>
#include <stdint.h>

enum { BMTE_TILE = 128, BMTE_ROW_BYTES = 64 };
typedef enum {
  BMTE_I8_TERNARY, BMTE_I4_TERNARY, BMTE_I8_BINARY, BMTE_I4_BINARY
} bitmte_mode;

/* Call the geometry helpers with a valid bitmte_mode. */
unsigned bitmte_a_bits(bitmte_mode mode);
unsigned bitmte_b_bits(bitmte_mode mode);
unsigned bitmte_k_slot(bitmte_mode mode);
unsigned bitmte_k_resident(bitmte_mode mode);
unsigned bitmte_slices(bitmte_mode mode);

/* One contiguous row; count must fill whole bytes. Returns 0 or -1 for invalid
 * mode/count/value. Buffers must be nonnull, large enough, and nonoverlapping.
 * Packers may partially write raw on error. Unpack rejects the ternary UB code.
 */
int bitmte_pack_a(uint8_t *raw, const int8_t *values, size_t count, bitmte_mode mode);
int bitmte_pack_b(uint8_t *raw, const int8_t *values, size_t count, bitmte_mode mode);
int bitmte_unpack_a(int8_t *values, const uint8_t *raw, size_t count, bitmte_mode mode);
int bitmte_unpack_b(int8_t *values, const uint8_t *raw, size_t count, bitmte_mode mode);

/* M-mode bare-metal initialization; minit does not zero matrix registers. */
void bitmte_init(void);

/* C[M][N] = A[M][K] * B[K][N], accumulated modulo 2^32 with msaten=0.
 * a: packed rows [M][K*a_bits/8]; bt: packed columns [N][K*b_bits/8].
 * M/N must be positive multiples of 128, K of k_resident(mode). All three
 * buffers are 64B aligned, mutually disjoint, physically contiguous and live
 * until return. Inputs stay immutable. C is overwritten, not accumulated.
 * Owns tr0..3, acc0..3, their configs, tile CSRs, msaten and sync0 exclusively.
 * Returns after matrix stores are visible, or -1 for invalid software arguments
 * before issuing matrix instructions. Does not validate buffer extents/data UB.
 */
int bitmte_gemm(uint32_t *c, const uint8_t *a, const uint8_t *bt,
                size_t m, size_t n, size_t k, bitmte_mode mode);

#endif
