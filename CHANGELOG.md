# Changelog

All notable changes to the TurboQuant project will be documented in this file.

## [0.3.1-stable] - 2026-03-31
- Corrected Ruff import sorting across core modules.
- Fixed TypeError in QJL adaptive tests by updating API calls.

## [0.3.0] - 2026-03-31
### Added
- **True 3-bit PolarQuant**: Implemented physical bit-packing (8x3-bit into 3x8-byte) for 5.3x - 6.0x compression of KV cache.
- **Cross-Layer KV Delta Compression**: New cache backend exploiting inter-layer correlation to achieve 11x-14x total compression via 1-bit signed deltas.
- **Speculative KV Prefill**: Draft-based prefill acceleration using ultra-compressed 1-bit sketches to predict KV states.
- **Temporal Expert Fusion**: On-the-fly SVD-based merging of rarely-used experts in MoE models to reduce VRAM footprint by ~20%.
- **Cross-Request Prefix Sharing**: Global KV cache manager for sharing common prefixes (system prompts) across concurrent server requests.
- **KV Watermarking**: HMAC-seeded LSB watermarking of KV scales for cryptographic attribution and anti-theft.
- **AsyncExpertLoader**: CUDA stream-based non-blocking expert transfers with double-buffering support.
- **Ultimate Benchmark**: New comprehensive performance and quality evaluation suite.

### Changed
- Replaced `_rotate` matmul with **Fast Walsh-Hadamard Transform (FWHT)** for $O(N \log N)$ orthogonalization on power-of-2 dimensions.
- Updated `QJLResidualCorrector` to support **norm-preserving encoding** and adaptive sketching.
- Optimized Lloyd-Max calibration using vectorized grid interpolation (10x faster).
- Migrated to `v0.3.0` core architecture with standardized `metadata.json` for persistence.

### Improved
- Memory usage reporting now reflects true packed storage rather than byte-padded containers.
- Enhanced telemetry with `hidden_io_percent` and `overall_compression_x`.

## [0.1.1] - 2026-03-27
- Fixes for PolarQuant int8 packing containers and QJL initialization.

## [0.1.0] - 2026-03-26
- Initial public release of core TurboQuant engine.
- Implementation of PolarQuant and QJL residuals.
