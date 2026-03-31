"""TurboQuant v0.3.0 — High-Performance KV-cache quantization for LLMs.

Up to 14× memory reduction with zero recall degradation.
Features: True 3-bit packing, Cross-layer Delta, Speculative Prefill, Expert Fusion.
"""

from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache
from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.qjl import QJLResidualCorrector, QJLConfig, AdaptiveQJLCorrector
from turboquant.core.cross_layer_kv_delta import CrossLayerDeltaConfig, CrossLayerKVDeltaCache
from turboquant.core.speculative_prefill import SpeculativePrefillEngine, SpeculativePrefillConfig
from turboquant.core.cross_request_kv import CrossRequestKVCache
from turboquant.core.kv_watermark import KVCacheWatermarker, WatermarkConfig
from turboquant.core.temporal_expert_fusion import TemporalExpertFusion, FusionConfig, ExpertUsageTracker

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
