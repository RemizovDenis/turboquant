"""TurboQuant v0.3.0 — High-Performance KV-cache quantization for LLMs.

Up to 14× memory reduction with zero recall degradation.
Features: True 3-bit packing, Cross-layer Delta, Speculative Prefill, Expert Fusion.
"""

from turboquant.core.cross_layer_kv_delta import CrossLayerDeltaConfig, CrossLayerKVDeltaCache
from turboquant.core.cross_request_kv import CrossRequestKVCache
from turboquant.core.kv_watermark import KVCacheWatermarker, WatermarkConfig
from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.qjl import AdaptiveQJLCorrector, QJLConfig, QJLResidualCorrector
from turboquant.core.speculative_prefill import SpeculativePrefillConfig, SpeculativePrefillEngine
from turboquant.core.temporal_expert_fusion import (
    ExpertUsageTracker,
    FusionConfig,
    TemporalExpertFusion,
)
from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

__version__ = "0.3.0"

__all__ = [
    "TurboQuantKVCache",
    "TurboQuantConfig",
    "CacheEntry",
    "PolarQuantizer",
    "QJLResidualCorrector",
    "QJLConfig",
    "AdaptiveQJLCorrector",
    "CrossLayerDeltaConfig",
    "CrossLayerKVDeltaCache",
    "SpeculativePrefillEngine",
    "SpeculativePrefillConfig",
    "CrossRequestKVCache",
    "KVCacheWatermarker",
    "WatermarkConfig",
    "TemporalExpertFusion",
    "FusionConfig",
    "ExpertUsageTracker",
    "__version__",
]
