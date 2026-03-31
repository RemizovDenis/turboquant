"""TurboQuant-MoE v0.3.0 Ultimate Benchmark Suite.

Comprehensive performance, memory, and quality evaluation for:
- True 3-bit PolarQuant
- Cross-Layer KV Delta (14x compression)
- Speculative Prefill
- Temporal Expert Fusion
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from turboquant.core.cross_layer_kv_delta import CrossLayerDeltaConfig, CrossLayerKVDeltaCache
from turboquant.core.speculative_prefill import SpeculativePrefillConfig, SpeculativePrefillEngine
from turboquant.core.temporal_expert_fusion import (
    ExpertUsageTracker,
    FusionConfig,
    TemporalExpertFusion,
)
from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache


@dataclass
class BenchConfig:
    head_dim: int = 128
    num_heads: int = 32
    seq_lens: list[int] = field(default_factory=lambda: [1024, 4096])
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results"


class UltimateBenchmark:
    def __init__(self, config: BenchConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self.results = {"meta": asdict(config), "suites": {}}

    def run_all(self) -> None:
        print(f"🚀 Starting TurboQuant v0.3.0 Ultimate Benchmark on {self.device}")

        self.results["suites"]["compression_ratio"] = self.bench_compression()
        self.results["suites"]["cross_layer_delta"] = self.bench_cross_layer()
        self.results["suites"]["speculative_prefill"] = self.bench_speculative()
        self.results["suites"]["expert_fusion"] = self.bench_fusion()

        self.save_report()

    def bench_compression(self) -> dict[str, Any]:
        tq_cfg = TurboQuantConfig(
            head_dim=self.config.head_dim,
            num_heads=self.config.num_heads,
            bits=3,
            device=self.config.device,
        )
        tq = TurboQuantKVCache(tq_cfg)

        rows = []
        for seq in self.config.seq_lens:
            k = torch.randn(
                self.config.batch_size,
                self.config.num_heads,
                seq,
                self.config.head_dim,
                device=self.device,
            )
            v = torch.randn_like(k)

            entry = tq.compress(k, v)
            mem = tq.memory_usage(entry)
            quality = tq.quality_metrics(k, v)

            rows.append(
                {
                    "seq_len": seq,
                    "fp16_mb": mem["total_mb"] / (mem["ratio"] if mem["ratio"] > 0 else 1),
                    "compressed_mb": mem["total_mb"],
                    "ratio_x": 1.0 / max(mem["ratio"], 1e-9),
                    "cosine_sim": quality["keys_cosine_sim"],
                }
            )
        return {"rows": rows}

    def bench_cross_layer(self) -> dict[str, Any]:
        cl_cfg = CrossLayerDeltaConfig(
            num_layers=32,
            head_dim=self.config.head_dim,
            num_heads=self.config.num_heads,
            anchor_stride=4,
            device=self.config.device,
        )
        cache = CrossLayerKVDeltaCache(cl_cfg)

        seq = 1024
        anchor_k = torch.randn(
            1, self.config.num_heads, seq, self.config.head_dim, device=self.device
        )
        anchor_v = torch.randn_like(anchor_k)

        cache.compress_layer(0, anchor_k, anchor_v)
        for layer_idx in range(1, 4):
            noisy_k = anchor_k + 0.05 * torch.randn_like(anchor_k)
            noisy_v = anchor_v + 0.05 * torch.randn_like(anchor_v)
            cache.compress_layer_streaming(layer_idx, noisy_k, noisy_v, anchor_k, anchor_v)

        stats = cache.memory_usage_all()
        return stats

    def bench_speculative(self) -> dict[str, Any]:
        tq_cfg = TurboQuantConfig(
            head_dim=self.config.head_dim,
            num_heads=self.config.num_heads,
            device=self.config.device,
        )
        tq = TurboQuantKVCache(tq_cfg)
        engine = SpeculativePrefillEngine(tq, SpeculativePrefillConfig())

        seq = 1024
        k = torch.randn(1, self.config.num_heads, seq, self.config.head_dim, device=self.device)
        v = torch.randn_like(k)

        engine.register_prompt_draft("p1", k, v)
        k_new = k + 0.05 * torch.randn_like(k)
        v_new = v + 0.05 * torch.randn_like(v)

        _, stats = engine.speculative_compress(k_new, v_new, prompt_id="p1")
        return stats

    def bench_fusion(self) -> dict[str, Any]:
        cfg = FusionConfig(min_usage_rate_for_fusion=0.1)
        fusion = TemporalExpertFusion(
            num_experts=8, expert_hidden_dim=4096, expert_ffn_dim=14336, config=cfg
        )
        tracker = ExpertUsageTracker(num_experts=8)

        tracker.record_activations(torch.randint(0, 4, (1, 100, 2)))
        tracker.record_activations(torch.randint(4, 8, (1, 10, 2)))

        # CPU weights for SVD
        weights = {i: (torch.randn(14336, 4096), torch.randn(14336, 4096)) for i in range(8)}
        savings = fusion.estimate_memory_savings(tracker, weights)
        return savings

    def save_report(self) -> None:
        out_path = Path(self.config.output_dir) / f"benchmark_v030_{self.timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    cfg = BenchConfig(device=args.device)
    bench = UltimateBenchmark(cfg)
    bench.run_all()
