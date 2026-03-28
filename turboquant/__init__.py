"""TurboQuant — Production-ready KV-cache quantization for LLMs.

4× memory reduction with zero recall degradation.
Reference: TurboQuant (arXiv 2504.19874).
"""

from turboquant.core.adaptive_bitwidth import (
    AdaptiveBitwidthConfig,
    AdaptiveBitwidthQuantizer,
    AdaptiveBitwithConfig,
    TokenImportanceClassifier,
)
from turboquant.core.cross_layer_kv import CrossLayerConfig, CrossLayerKVCache
from turboquant.core.markov_prefetch import MarkovPrefetchConfig, MarkovTrajectoryPredictor
from turboquant.core.nash_router import GameTheoreticRouter, NashRouterConfig
from turboquant.core.pid_vram import PIDConfig, VRAM_PID_Controller
from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.qjl import QJLResidualCorrector
from turboquant.core.semantic_eviction import SemanticEvictionConfig, SemanticKVEviction
from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

__version__ = "0.1.2"

__all__ = [
    "PolarQuantizer",
    "QJLResidualCorrector",
    "AdaptiveBitwithConfig",
    "AdaptiveBitwidthConfig",
    "AdaptiveBitwidthQuantizer",
    "TokenImportanceClassifier",
    "CrossLayerConfig",
    "CrossLayerKVCache",
    "GameTheoreticRouter",
    "NashRouterConfig",
    "MarkovTrajectoryPredictor",
    "MarkovPrefetchConfig",
    "PIDConfig",
    "VRAM_PID_Controller",
    "SemanticEvictionConfig",
    "SemanticKVEviction",
    "TurboQuantKVCache",
    "TurboQuantConfig",
    "CacheEntry",
    "__version__",
]
