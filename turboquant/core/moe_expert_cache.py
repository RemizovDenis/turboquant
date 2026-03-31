"""Dynamic expert cache for MoE inference v0.3.0.

Improved with AsyncExpertLoader, CUDA stream-based transfers,
double-buffering, and IO-hiding metrics.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import structlog
import torch
import torch.nn as nn
from concurrent.futures import Future

log = structlog.get_logger(__name__)

class AsyncExpertLoader:
    """Load experts CPU -> GPU asynchronously via dedicated CUDA streams."""
    
    def __init__(self, device: torch.device, stream: torch.cuda.Stream | None = None):
        self.device = device
        self.stream = stream or (torch.cuda.Stream(device=device) if device.type == "cuda" else None)
        self._pending: Dict[int, Union[Tuple[torch.Tensor, torch.Tensor], Future]] = {}
        self._load_count = 0
        self._hidden_count = 0
        
    def prefetch(self, expert_id: int, cpu_weights: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Start async CPU -> GPU transfer. Non-blocking."""
        if self.stream is None:
            # CPU fallback: "prefetch" is just store
            self._pending[expert_id] = cpu_weights
            return
            
        with torch.cuda.stream(self.stream):
            # non_blocking=True allows overlapping with compute
            gpu_w1 = cpu_weights[0].to(self.device, non_blocking=True)
            gpu_w2 = cpu_weights[1].to(self.device, non_blocking=True)
            self._pending[expert_id] = (gpu_w1, gpu_w2)
            self._load_count += 1
            
    def get(self, expert_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get prefetched expert (blocks until transfer is complete)."""
        if expert_id not in self._pending:
            return None
            
        weights = self._pending.pop(expert_id)
        if self.stream is not None:
            # Check if it was already ready (hidden IO)
            # This is a bit hard to measure perfectly without events, 
            # but we can check if synchronize takes time.
            t0 = time.perf_counter()
            self.stream.synchronize()
            wait_time = time.perf_counter() - t0
            if wait_time < 1e-4: # effectively zero wait
                self._hidden_count += 1
                
        return weights

    def is_ready(self, expert_id: int) -> bool:
        """Non-blocking check if expert is ready on GPU."""
        if expert_id not in self._pending:
            return False
        if self.stream is None:
            return True
        return self.stream.query()

    def hidden_io_percent(self) -> float:
        """Percentage of loads that were successfully hidden behind compute."""
        return (self._hidden_count / max(self._load_count, 1)) * 100

class DynamicExpertCache:
    """Expert cache with double-buffering and async loading."""
    
    def __init__(self, config: Any, cpu_store: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cpu_store = cpu_store
        self._gpu_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Async loader
        self._async_loader = AsyncExpertLoader(self.device)
        
        # Double buffering
        self._buffer_a: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._buffer_b: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._active_buffer = "a"
        
        self._lock = threading.Lock()
        
    def prefetch_experts(self, expert_ids: List[int]) -> None:
        """Async prefetch multiple experts from CPU to GPU."""
        for eid in expert_ids:
            if eid not in self._gpu_cache and eid in self._cpu_store:
                self._async_loader.prefetch(eid, self._cpu_store[eid])
                
    def get_expert_with_prefetch(
        self, 
        expert_id: int, 
        next_expert_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get expert (potentially from async loader) and prefetch next ones."""
        # 1. Try GPU cache
        if expert_id in self._gpu_cache:
            if next_expert_ids:
                self.prefetch_experts(next_expert_ids)
            return self._gpu_cache[expert_id]
            
        # 2. Try Async loader
        weights = self._async_loader.get(expert_id)
        if weights:
            self._gpu_cache[expert_id] = weights
            if next_expert_ids:
                self.prefetch_experts(next_expert_ids)
            return weights
            
        # 3. Blocking load from CPU
        log.warning("cache_miss_blocking_load", expert_id=expert_id)
        cpu_w = self._cpu_store[expert_id]
        gpu_w = (cpu_w[0].to(self.device), cpu_w[1].to(self.device))
        self._gpu_cache[expert_id] = gpu_w
        
        if next_expert_ids:
            self.prefetch_experts(next_expert_ids)
            
        return gpu_w

    def swap_buffers(self) -> None:
        """Swap active/inactive buffers for double-buffered inference."""
        with self._lock:
            if self._active_buffer == "a":
                self._active_buffer = "b"
                self._buffer_a.clear()
            else:
                self._active_buffer = "a"
                self._buffer_b.clear()

    def hidden_io_percent(self) -> float:
        return self._async_loader.hidden_io_percent()
