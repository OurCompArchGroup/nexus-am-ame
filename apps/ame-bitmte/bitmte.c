#include "bitmte.h"
#include <string.h>

unsigned bitmte_a_bits(bitmte_mode mode) {
  return mode == BMTE_I4_TERNARY || mode == BMTE_I4_BINARY ? 4 : 8;
}
unsigned bitmte_b_bits(bitmte_mode mode) {
  return mode == BMTE_I8_BINARY || mode == BMTE_I4_BINARY ? 1 : 2;
}
unsigned bitmte_k_slot(bitmte_mode mode) { return 512 / bitmte_a_bits(mode); }
unsigned bitmte_k_resident(bitmte_mode mode) { return 512 / bitmte_b_bits(mode); }
unsigned bitmte_slices(bitmte_mode mode) { return bitmte_a_bits(mode) / bitmte_b_bits(mode); }

int bitmte_pack_a(uint8_t *raw, const int8_t *values, size_t count, bitmte_mode mode) {
  if ((unsigned)mode > BMTE_I4_BINARY || !raw || !values) return -1;
  unsigned bits = bitmte_a_bits(mode), lanes = 8 / bits;
  if (count % lanes) return -1;
  memset(raw, 0, count / lanes);
  for (size_t k = 0; k < count; ++k) {
    int value = values[k];
    if (bits == 4 && (value < -8 || value > 7)) return -1;
    raw[k / lanes] |= (uint8_t)(((unsigned)value & ((1u << bits) - 1)) << (bits * (k % lanes)));
  }
  return 0;
}

int bitmte_pack_b(uint8_t *raw, const int8_t *values, size_t count, bitmte_mode mode) {
  if ((unsigned)mode > BMTE_I4_BINARY || !raw || !values) return -1;
  unsigned bits = bitmte_b_bits(mode), lanes = 8 / bits;
  if (count % lanes) return -1;
  memset(raw, 0, count / lanes);
  for (size_t k = 0; k < count; ++k) {
    int value = values[k];
    if (value < (bits == 2 ? -1 : 0) || value > 1) return -1;
    unsigned encoded = value < 0 ? 3 : (unsigned)value;
    raw[k / lanes] |= (uint8_t)(encoded << (bits * (k % lanes)));
  }
  return 0;
}

int bitmte_unpack_a(int8_t *values, const uint8_t *raw, size_t count, bitmte_mode mode) {
  if ((unsigned)mode > BMTE_I4_BINARY || !raw || !values) return -1;
  unsigned bits = bitmte_a_bits(mode), lanes = 8 / bits;
  if (count % lanes) return -1;
  for (size_t k = 0; k < count; ++k) {
    unsigned field = (raw[k / lanes] >> (bits * (k % lanes))) & ((1u << bits) - 1);
    values[k] = (int8_t)(field < (1u << (bits - 1)) ? (int)field : (int)field - (1 << bits));
  }
  return 0;
}

int bitmte_unpack_b(int8_t *values, const uint8_t *raw, size_t count, bitmte_mode mode) {
  if ((unsigned)mode > BMTE_I4_BINARY || !raw || !values) return -1;
  unsigned bits = bitmte_b_bits(mode), lanes = 8 / bits;
  if (count % lanes) return -1;
  for (size_t k = 0; k < count; ++k) {
    unsigned field = (raw[k / lanes] >> (bits * (k % lanes))) & ((1u << bits) - 1);
    if (bits == 2 && field == 2) return -1;
    values[k] = field == 3 ? -1 : (int8_t)field;
  }
  return 0;
}
