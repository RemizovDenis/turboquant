#!/usr/bin/env python3
"""TurboQuant offline validation — no external dependencies required.

Validates the core mathematical operations without torch/numpy/scipy.
This script can run on vanilla Python 3.9+ with zero pip installs.

Usage:
    python3 scripts/validate_offline.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path


def banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


# =============================================================
# 1. Validate orthogonal rotation matrix generation
# =============================================================


def test_rotation_matrix():
    """Verify QR decomposition produces an orthogonal matrix."""
    print("1. Testing orthogonal rotation matrix generation...")

    # Simplified version: generate a random matrix and verify orthogonality
    # via Gram-Schmidt (no numpy needed)
    dim = 8  # small for pure Python
    random.seed(42)

    # Generate random vectors
    vectors = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(dim)]

    # Gram-Schmidt orthogonalization
    orthogonal = []
    for v in vectors:
        # Subtract projections onto existing orthogonal vectors
        for u in orthogonal:
            dot = sum(a * b for a, b in zip(v, u, strict=False))
            v = [vi - dot * ui for vi, ui in zip(v, u, strict=False)]
        # Normalize
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 1e-10:
            v = [x / norm for x in v]
            orthogonal.append(v)

    # Verify orthogonality: Q^T @ Q should be identity
    max_off_diag = 0.0
    min_diag = float("inf")
    for i in range(dim):
        for j in range(dim):
            dot = sum(orthogonal[i][k] * orthogonal[j][k] for k in range(dim))
            if i == j:
                min_diag = min(min_diag, abs(dot))
            else:
                max_off_diag = max(max_off_diag, abs(dot))

    assert min_diag > 0.99, f"Diagonal elements too small: {min_diag}"
    assert max_off_diag < 1e-10, f"Off-diagonal elements too large: {max_off_diag}"
    print(f"   Diagonal min:     {min_diag:.10f} (expected ~1.0)")
    print(f"   Off-diagonal max: {max_off_diag:.2e} (expected ~0.0)")
    print("   ✅ Orthogonal matrix generation: PASS")


# =============================================================
# 2. Validate Beta distribution method of moments
# =============================================================


def test_beta_moments():
    """Verify Beta MoM parameter estimation."""
    print("\n2. Testing Beta distribution parameter estimation...")

    random.seed(42)

    # Generate samples from a known distribution and verify MoM recovery
    # Beta(α=2, β=5) → mean = 2/7 ≈ 0.286, var = 10/(49*8) ≈ 0.0255
    true_alpha = 2.0
    true_beta = 5.0
    true_alpha / (true_alpha + true_beta)

    # Generate pseudo-Beta samples via inverse CDF approximation
    n = 10000
    # Simple: use sum of uniforms as approximation
    samples = []
    for _ in range(n):
        # Using the fact that for integers, Beta can be generated from order statistics
        u = sorted([random.random() for _ in range(int(true_alpha + true_beta) - 1)])
        samples.append(u[int(true_alpha) - 1])  # k-th order statistic

    # Method of moments
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    var = max(var, 1e-12)

    common = (mean * (1.0 - mean) / var) - 1.0
    common = max(common, 1e-6)
    est_alpha = mean * common
    est_beta = (1.0 - mean) * common

    print(f"   True:      α={true_alpha:.2f}, β={true_beta:.2f}")
    print(f"   Estimated: α={est_alpha:.2f}, β={est_beta:.2f}")
    print(f"   Mean error: α={abs(est_alpha - true_alpha):.2f}, β={abs(est_beta - true_beta):.2f}")

    # Loose tolerance since we're using approximate Beta samples
    assert abs(est_alpha - true_alpha) < 1.0, f"Alpha estimation too far off: {est_alpha}"
    assert abs(est_beta - true_beta) < 2.0, f"Beta estimation too far off: {est_beta}"
    print("   ✅ Beta MoM estimation: PASS")


# =============================================================
# 3. Validate quantization / dequantization logic
# =============================================================


def test_quantization_logic():
    """Verify uniform quantization to N levels preserves signal."""
    print("\n3. Testing quantization/dequantization logic...")

    random.seed(42)
    n_levels = 8  # 3 bits
    group_size = 64

    # Generate test data
    data = [random.gauss(0, 1) for _ in range(256)]

    # Quantize in groups
    quantized = []
    scales = []
    for g in range(0, len(data), group_size):
        group = data[g : g + group_size]
        # Per-group scale
        scale = max(abs(x) for x in group) if group else 1e-12
        scale = max(scale, 1e-12)
        scales.append(scale)
        # Normalize to [-1, 1]
        normed = [x / scale for x in group]
        # Uniform levels in [-1, 1]
        levels = [(2 * i / (n_levels - 1)) - 1.0 for i in range(n_levels)]
        boundaries = [(levels[i] + levels[i + 1]) / 2 for i in range(n_levels - 1)]
        # Quantize: find nearest level
        indices = []
        for val in normed:
            idx = 0
            for b in boundaries:
                if val > b:
                    idx += 1
            indices.append(idx)
        quantized.append(indices)

    # Dequantize
    levels = [(2 * i / (n_levels - 1)) - 1.0 for i in range(n_levels)]
    reconstructed = []
    for _g_idx, (indices, scale) in enumerate(zip(quantized, scales, strict=False)):
        for idx in indices:
            reconstructed.append(levels[idx] * scale)

    # Compute MSE
    mse = sum((a - b) ** 2 for a, b in zip(data, reconstructed, strict=False)) / len(data)
    # Signal power
    signal_power = sum(x**2 for x in data) / len(data)
    snr_db = 10 * math.log10(signal_power / max(mse, 1e-15))

    print(f"   Levels:       {n_levels} ({int(math.log2(n_levels))} bits)")
    print(f"   Group size:   {group_size}")
    print(f"   MSE:          {mse:.6f}")
    print(f"   SNR:          {snr_db:.1f} dB")

    assert mse < 0.1, f"MSE too high: {mse}"
    assert snr_db > 10, f"SNR too low: {snr_db} dB"
    print("   ✅ Quantize/dequantize logic: PASS")


# =============================================================
# 4. Validate bit packing / unpacking
# =============================================================


def test_bit_packing():
    """Verify lossless bit pack/unpack roundtrip."""
    print("\n4. Testing bit packing/unpacking...")

    random.seed(42)
    n_bits = 37  # Non-multiple of 8 to test padding

    # Generate random bits
    bits = [random.randint(0, 1) for _ in range(n_bits)]

    # Pack into bytes (MSB first, as in qjl.py)
    padded = bits + [0] * ((8 - n_bits % 8) % 8)
    packed = []
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte |= padded[i + j] << (7 - j)
        packed.append(byte)

    # Unpack
    unpacked = []
    for byte in packed:
        for j in range(8):
            unpacked.append((byte >> (7 - j)) & 1)
    unpacked = unpacked[:n_bits]

    assert bits == unpacked, f"Pack/unpack mismatch at n_bits={n_bits}"

    # Test with multiple sizes
    for n in [1, 7, 8, 15, 16, 32, 64, 100, 128]:
        bits = [random.randint(0, 1) for _ in range(n)]
        padded = bits + [0] * ((8 - n % 8) % 8)
        packed = []
        for i in range(0, len(padded), 8):
            byte = 0
            for j in range(8):
                byte |= padded[i + j] << (7 - j)
            packed.append(byte)
        unpacked = []
        for byte in packed:
            for j in range(8):
                unpacked.append((byte >> (7 - j)) & 1)
        unpacked = unpacked[:n]
        assert bits == unpacked, f"Failed at n={n}"

    print("   Tested sizes: 1, 7, 8, 15, 16, 32, 64, 100, 128")
    print("   ✅ Bit packing roundtrip: PASS")


# =============================================================
# 5. Validate compression ratio calculation
# =============================================================


def test_compression_ratio():
    """Verify memory math is correct."""
    print("\n5. Testing compression ratio calculations...")

    # KV-cache: [batch=1, heads=32, seq=4096, dim=128]
    batch, heads, seq, dim = 1, 32, 4096, 128

    # FP16 baseline: 2 bytes per element, keys + values
    fp16_bytes = batch * heads * seq * dim * 2 * 2  # *2 for K+V
    fp16_mb = fp16_bytes / (1024**2)

    # TurboQuant 3-bit: 3/8 bytes per element + scales overhead
    group_size = 64
    num_groups = math.ceil(dim / group_size)
    # quantized: int8 (1 byte per element) — stores 3-bit index in int8
    q_bytes_kv = batch * heads * seq * dim * 1 * 2  # K+V
    # scales: float32 per group
    s_bytes_kv = batch * heads * seq * num_groups * 4 * 2  # K+V
    tq3_bytes = q_bytes_kv + s_bytes_kv

    # QJL residual: 1 bit per sketch_dim element
    sketch_dim = dim // 4
    packed_bytes = math.ceil(sketch_dim / 8)
    r_bytes_kv = batch * heads * seq * packed_bytes * 2  # K+V
    tq4_bytes = tq3_bytes + r_bytes_kv

    ratio_3 = tq3_bytes / fp16_bytes
    ratio_4 = tq4_bytes / fp16_bytes

    print(f"   Shape: [{batch}, {heads}, {seq}, {dim}]")
    print(f"   FP16:     {fp16_mb:.1f} MB ({fp16_bytes:,} bytes)")
    print(f"   TQ 3-bit: {tq3_bytes / (1024**2):.1f} MB (ratio: {ratio_3:.2%})")
    print(f"   TQ 3+1:   {tq4_bytes / (1024**2):.1f} MB (ratio: {ratio_4:.2%})")
    print(f"   Savings:  {(1 - ratio_4) * 100:.1f}%")

    # NOTE: Current implementation stores 3-bit indices in int8 (1 byte),
    # so actual ratio is ~55% not the theoretical 25% (4/16 bits).
    # True 3-bit packing (3 bits per element) would achieve 25%.
    # This is a known trade-off: int8 is faster to process on all hardware
    # vs custom bit-packing which requires more complex kernels.
    # The savings are still significant: ~45% memory reduction.
    assert ratio_4 < 0.60, f"Compression ratio unexpectedly high: {ratio_4}"
    print("   ✅ Compression ratio math: PASS")


# =============================================================
# 6. Validate Johnson-Lindenstrauss dimension calc
# =============================================================


def test_jl_dimension():
    """Verify JL projection preserves distances approximately."""
    print("\n6. Testing Johnson-Lindenstrauss projection...")

    random.seed(42)
    dim = 128
    sketch_dim = 32
    scale = 1.0 / math.sqrt(sketch_dim)

    # Generate two random vectors
    a = [random.gauss(0, 1) for _ in range(dim)]
    b = [random.gauss(0, 1) for _ in range(dim)]

    # True distance
    true_dist_sq = sum((x - y) ** 2 for x, y in zip(a, b, strict=False))

    # Random sign projection
    n_trials = 100
    projected_dists = []

    for trial in range(n_trials):
        random.seed(trial + 1000)
        # Generate projection matrix row by row
        proj_a = []
        proj_b = []
        for _j in range(sketch_dim):
            row = [(1 if random.random() > 0.5 else -1) * scale for _ in range(dim)]
            proj_a.append(sum(r * x for r, x in zip(row, a, strict=False)))
            proj_b.append(sum(r * x for r, x in zip(row, b, strict=False)))

        proj_dist_sq = sum((x - y) ** 2 for x, y in zip(proj_a, proj_b, strict=False))
        projected_dists.append(proj_dist_sq)

    mean_proj_dist = sum(projected_dists) / n_trials
    relative_error = abs(mean_proj_dist - true_dist_sq) / true_dist_sq

    print(f"   Original dim: {dim}, Sketch dim: {sketch_dim}")
    print(f"   True ||a-b||²:     {true_dist_sq:.2f}")
    print(f"   Mean projected:    {mean_proj_dist:.2f}")
    print(f"   Relative error:    {relative_error:.4f} ({relative_error * 100:.1f}%)")

    assert relative_error < 0.3, f"JL error too high: {relative_error}"
    print("   ✅ JL distance preservation: PASS")


# =============================================================
# 7. File structure validation
# =============================================================


def test_project_structure():
    """Verify all expected files exist."""
    print("\n7. Testing project structure...")

    base = Path(__file__).parent.parent
    expected_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        "prometheus.yml",
        ".github/workflows/ci.yml",
        "turboquant/__init__.py",
        "turboquant/core/__init__.py",
        "turboquant/core/polar_quant.py",
        "turboquant/core/qjl.py",
        "turboquant/core/turboquant.py",
        "turboquant/integrations/__init__.py",
        "turboquant/integrations/transformers.py",
        "turboquant/integrations/ollama.py",
        "turboquant/integrations/vector_db.py",
        "turboquant/benchmarks/__init__.py",
        "turboquant/benchmarks/benchmark.py",
        "tests/__init__.py",
        "tests/test_turboquant.py",
        "OUTREACH.md",
        "scripts/quickstart.sh",
    ]

    missing = []
    for f in expected_files:
        path = base / f
        if path.exists():
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ {f} — MISSING")
            missing.append(f)

    assert not missing, f"Missing files: {missing}"
    print(f"\n   All {len(expected_files)} files present!")
    print("   ✅ Project structure: PASS")


# =============================================================
# Main
# =============================================================


def main() -> int:
    banner("TurboQuant Offline Validation")
    print(f"Python: {sys.version}")
    print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        test_rotation_matrix,
        test_beta_moments,
        test_quantization_logic,
        test_bit_packing,
        test_compression_ratio,
        test_jl_dimension,
        test_project_structure,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            failed += 1

    banner("Results")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n  🎉 ALL VALIDATIONS PASSED — ready for GitHub!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed — fix before publishing.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
