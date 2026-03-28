"""TurboQuant — Production-ready KV-cache quantization for LLMs.

4× memory reduction with zero recall degradation.
Reference: TurboQuant (arXiv 2504.19874).
"""

from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.qjl import QJLResidualCorrector
from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

__version__ = "0.1.1"

__all__ = [
    "PolarQuantizer",
    "QJLResidualCorrector",
    "TurboQuantKVCache",
    "TurboQuantConfig",
    "CacheEntry",
    "__version__",
]
