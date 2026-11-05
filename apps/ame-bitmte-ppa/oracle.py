#!/usr/bin/env python3
"""Build the complete integer GEMM oracle independently of packing/scheduling.

For a signed four-bit value a, a = p0 + 2*p1 + 4*p2 - 8*p3.
B is binary; intersections and population counts therefore implement its exact
integer dot product. The complete packed A/B fixture is embedded at build time.
No third-party package or matrix emulator is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def mix32(value):
    value &= 0xffffffff
    value ^= value >> 16
    value = value * 0x7feb352d & 0xffffffff
    value ^= value >> 15
    value = value * 0x846ca68b & 0xffffffff
    return value ^ (value >> 16)


def fnv_step(state, value):
    return ((state ^ value) * 16777619) & 0xffffffff


def bit_planes(row, bits):
    packed = [bytearray((len(row) + 7) // 8) for _ in range(bits)]
    for k, value in enumerate(row):
        for bit in range(bits):
            packed[bit][k // 8] |= ((value >> bit) & 1) << (k % 8)
    return [int.from_bytes(plane, 'little') for plane in packed]


def dot(planes, mask):
    return ((planes[0] & mask).bit_count()
            + 2 * (planes[1] & mask).bit_count()
            + 4 * (planes[2] & mask).bit_count()
            - 8 * (planes[3] & mask).bit_count())


def generate(m, n, k, seed, check_scalar=False):
    a = [[(mix32(i * 65537 + x + seed) & 15) - 8 for x in range(k)]
         for i in range(m)]
    b = [[mix32(j * 104729 + x + seed + 17) & 1 for x in range(k)]
         for j in range(n)]
    a_planes = [bit_planes(row, 4) for row in a]
    b_masks = [bit_planes(row, 1)[0] for row in b]
    expected = [dot(planes, mask) for planes in a_planes for mask in b_masks]
    checks = {'all_scalar_outputs': 0, 'single_lane_combinations': 0}
    if check_scalar:
        for av in range(-8, 8):
            for bv in range(2):
                assert dot(bit_planes([av], 4), bv) == av * bv
                checks['single_lane_combinations'] += 1
        for i, arow in enumerate(a):
            for j, brow in enumerate(b):
                scalar = sum(av * bv for av, bv in zip(arow, brow))
                if scalar != expected[i * n + j]:
                    raise ValueError(f'bit-plane/scalar mismatch at ({i},{j})')
                checks['all_scalar_outputs'] += 1
    raw_a = bytes(value & 255 for row in a for value in row)
    raw_b = bytes(value for row in b for value in row)
    input_hash = 2166136261
    for value in raw_a + raw_b:
        input_hash = fnv_step(input_hash, value)
    result_hash = 2166136261
    for value in expected:
        result_hash = fnv_step(result_hash, value & 0xffffffff)
    raw_expected = b''.join(struct.pack('<i', value) for value in expected)
    return expected, {
        'schema': 'ame-bitmte-ppa-oracle-v1',
        'm': m, 'n': n, 'k': k, 'seed': seed,
        'method': 'exact signed-four-bit A planes intersect binary B mask',
        'numeric_a_sha256': hashlib.sha256(raw_a).hexdigest(),
        'numeric_b_sha256': hashlib.sha256(raw_b).hexdigest(),
        'expected_int32_le_sha256': hashlib.sha256(raw_expected).hexdigest(),
        'input_hash': f'{input_hash:08x}',
        'numerical_hash': f'{result_hash:08x}',
        'expected_elements': len(expected),
        'minimum_output': min(expected), 'maximum_output': max(expected),
        'independent_checks': checks,
    }



MODES = {
    'i8-i8': (0, 8, 8), 'bf16-bf16': (1, 16, 16),
    'i8-fp2pack4': (2, 8, 2), 'i4-fp2pack4': (3, 4, 2),
    'i8-i1pack8': (4, 8, 1), 'i4-i1pack8': (5, 4, 1),
}


def packed_inputs(m, n, k, seed, mode):
    """Pack the frozen numerical values in each mode's native lane order."""
    _, a_bits, b_bits = MODES[mode]
    arrays = []
    for rows, bits, is_a in ((m, a_bits, True), (n, b_bits, False)):
        stride = k * bits // 8
        data = bytearray(rows * stride)
        for row in range(rows):
            for lane in range(k):
                value = ((mix32(row * 65537 + lane + seed) & 15) - 8 if is_a
                         else mix32(row * 104729 + lane + seed + 17) & 1)
                if bits == 16:
                    # Every value in the common domain is exactly BF16.
                    offset = row * stride + 2 * lane
                    data[offset:offset + 2] = struct.pack('<f', float(value))[2:]
                else:
                    offset = row * stride + lane * bits // 8
                    data[offset] |= (value & ((1 << bits) - 1)) << (lane * bits % 8)
        arrays.append(bytes(data))
    return arrays


def append_packed_array(lines, name, data, rows, stride, width_macro):
    dimension = 'M' if name == 'packed_a' else 'N'
    lines.append(f'static uint8_t {name}[PPA_{dimension}]'
                 f'[PPA_K * {width_macro} / 8u] __attribute__((aligned(64))) = {{')
    for row in range(rows):
        lines.append('  {')
        raw = data[row * stride:(row + 1) * stride]
        lines.extend('    ' + ', '.join(f'0x{value:02x}' for value in raw[start:start + 16]) + ','
                     for start in range(0, stride, 16))
        lines.append('  },')
    lines.append('};')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('m', 'n', 'k'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=lambda x: int(x, 0), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-scalar', action='store_true')
    parser.add_argument('--expected-input-hash')
    parser.add_argument('--expected-result-hash')
    args = parser.parse_args()
    if (args.m <= 0 or args.n <= 0 or args.m % 128 or args.n % 128 or
            args.k <= 0 or args.k % 512 or args.k > 2097152):
        parser.error('M/N must be positive multiples of128; K a positive multiple of512 <=2097152')
    expected, metadata = generate(args.m, args.n, args.k, args.seed, args.check_scalar)
    for actual, wanted in ((metadata['input_hash'], args.expected_input_hash),
                           (metadata['numerical_hash'], args.expected_result_hash)):
        if wanted is not None and actual != wanted:
            raise ValueError(f'existing oracle identity mismatch: got {actual}, expected {wanted}')
    mode_code, a_bits, b_bits = MODES[args.mode]
    raw_a, raw_b = packed_inputs(args.m, args.n, args.k, args.seed, args.mode)
    metadata.update(fixture_schema='ame-bitmte-ppa-static-fixture-v1',
                    mode=args.mode, a_bits=a_bits, b_bits=b_bits,
                    a_stride=args.k * a_bits // 8, b_stride=args.k * b_bits // 8,
                    packed_a_bytes=len(raw_a), packed_b_bytes=len(raw_b),
                    packed_a_sha256=hashlib.sha256(raw_a).hexdigest(),
                    packed_b_sha256=hashlib.sha256(raw_b).hexdigest(),
                    output_bytes=args.m * args.n * 4, output_initial_value=0,
                    input_hash_source='host_fixture')
    lines = ['/* Complete host-loaded input fixture and integer reference. */', '#pragma once',
             f'_Static_assert(PPA_MODE == {mode_code}, "fixture mode mismatch");',
             f'_Static_assert(PPA_M == {args.m} && PPA_N == {args.n} && PPA_K == {args.k}, "fixture shape mismatch");',
             f'_Static_assert(PPA_SEED == UINT32_C({args.seed}), "fixture seed mismatch");',
             f'#define PPA_ORACLE_INPUT_HASH UINT32_C(0x{metadata["input_hash"]})',
             f'#define PPA_ORACLE_RESULT_HASH UINT32_C(0x{metadata["numerical_hash"]})',
             f'static const int32_t ppa_expected[{len(expected)}] = {{']
    lines.extend('  ' + ', '.join(str(value) for value in expected[start:start + 8]) + ','
                 for start in range(0, len(expected), 8))
    lines.append('};')
    append_packed_array(lines, 'packed_a', raw_a, args.m, args.k * a_bits // 8, 'A_BITS')
    append_packed_array(lines, 'packed_b', raw_b, args.n, args.k * b_bits // 8, 'B_BITS')
    header = ('\n'.join(lines) + '\n').encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_bytes(header)
    temporary.replace(args.out)
    metadata.update(oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    header_sha256=hashlib.sha256(header).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'ORACLE_READY shape={args.m}x{args.n}x{args.k} '
          f'input={metadata["input_hash"]} result={metadata["numerical_hash"]} '
          f'scalar_checked={metadata["independent_checks"]["all_scalar_outputs"]}')


if __name__ == '__main__':
    main()
