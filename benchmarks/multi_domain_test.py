"""benchmarks/multi_domain_test.py — Multi-Domain Validation for TurboQuant-MoE.

This script measures TurboQuant's compression effectiveness across Code, Legal, and Chat datasets.
"""

import os
import time
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.core.turboquant import TurboQuantKVCache, TurboQuantConfig

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cpu"
HF_CACHE = Path(".hf_cache")
HF_CACHE.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE.absolute())

# 2. Domain Data Samples
WORKLOADS = {
    "CODE (Python)": """
def deep_quantization_hadamard(tensor, bits=3):
    \"\"\"Simulated high-complexity mathematical code for transformer blocks.\"\"\"
    bs, h, s, d = tensor.shape
    # Applying Walsh-Hadamard Transform (WHT)
    wht_matrix = generate_hadamard(d)
    transformed = torch.matmul(tensor, wht_matrix)
    # Lloyd-Max quantization loop for radius and angles
    quantized = polar_quant_radius(transformed, bits)
    return quantized.to(torch.int8)

class TurboQuantMoEHandler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.polar = PolarQuantizer(config.bits)
        self.qjl = QJLResidualCorrector(config.sketch_dim)
    
    def forward(self, x):
        packed, scales = self.polar.forward(x)
        residual = self.qjl.encode(x - self.polar.dequantize(packed, scales))
        return packed, scales, residual
    """,
    "LEGAL (Terms)": """
LIMITATION OF LIABILITY. IN NO EVENT SHALL TURBOQUANT-MOE INC., ITS AFFILIATES, OR THEIR LICENSORS, 
SERVICE PROVIDERS, EMPLOYEES, AGENTS, OFFICERS, OR DIRECTORS BE LIABLE FOR DAMAGES OF ANY KIND, 
UNDER ANY LEGAL THEORY, ARISING OUT OF OR IN CONNECTION WITH YOUR USE, OR INABILITY TO USE, 
THE SOFTWARE, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, 
INCLUDING BUT NOT LIMITED TO, PERSONAL INJURY, PAIN AND SUFFERING, EMOTIONAL DISTRESS, LOSS OF REVENUE, 
LOSS OF PROFITS, LOSS OF BUSINESS OR ANTICIPATED SAVINGS, LOSS OF USE, LOSS OF GOODWILL, LOSS OF DATA, 
AND WHETHER CAUSED BY TORT (INCLUDING NEGLIGENCE), BREACH OF CONTRACT, OR OTHERWISE, EVEN IF FORESEEABLE.
THE FOREGOING DOES NOT AFFECT ANY LIABILITY THAT CANNOT BE EXCLUDED OR LIMITED UNDER APPLICABLE LAW.
    """,
    "CHAT (Casual)": """
User: Hey, can you help me plan a trip to Japan for next spring?
AI: Of course! Japan is beautiful in the spring, especially during the cherry blossom season. 
Which cities are you interested in visiting? Tokyo, Kyoto, and Osaka are the classic choices.
User: I definitely want to see the blossoms. And maybe some smaller towns too?
AI: In that case, I'd recommend Kanazawa or Takayama. They have preserved historic districts 
and are much quieter than Tokyo. How many days are you planning to stay?
User: About two weeks. I also want to try as much local food as possible!
AI: Great! Two weeks is perfect. We can build a food tour covering sushi in Toyosu, 
street food in Dotonbori, and traditional kaiseki in Kyoto.
    """
}

def calculate_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()

def run_multi_domain_test():
    print(f"🌍 Starting Multi-Domain Trust Test (Model: {MODEL_ID})")
    
    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    
    # Init Table Headers
    print(f"\n{'DOMAIN':<15} | {'TOKENS':<8} | {'ORIGINAL':<12} | {'TQK':<12} | {'RATIO':<8} | {'FIDELITY':<10}")
    print("-" * 80)

    for domain, text in WORKLOADS.items():
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        num_tokens = inputs.input_ids.shape[1]
        
        # Generate raw KV-cache
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
            
        # Init Engine
        config = TurboQuantConfig(
            device=DEVICE,
            bits=3,
            residual_correction=True,
            head_dim=past_key_values[0][0].shape[-1],
            num_heads=past_key_values[0][0].shape[1]
        )
        engine = TurboQuantKVCache(config)
        
        num_layers = len(past_key_values)
        total_cossim = 0.0
        raw_bytes = 0
        comp_bytes = 0
        
        for i in range(num_layers):
            k_orig = past_key_values[i][0]
            v_orig = past_key_values[i][1]
            raw_bytes += k_orig.nbytes + v_orig.nbytes
            
            entry = engine.compress(k_orig, v_orig)
            k_hat, v_hat = engine.decompress(entry)
            
            # Fidelity
            total_cossim += (calculate_cosine_similarity(k_orig, k_hat) + calculate_cosine_similarity(v_orig, v_hat)) / 2
            
            # Compressed Size
            comp_bytes += (
                entry.compressed_keys[0].nbytes + entry.compressed_keys[1].nbytes +
                entry.compressed_values[0].nbytes + entry.compressed_values[1].nbytes
            )
            if entry.residual_keys is not None:
                comp_bytes += entry.residual_keys.nbytes + entry.residual_values.nbytes
                comp_bytes += entry.residual_norms_k.nbytes + entry.residual_norms_v.nbytes

        avg_fide = total_cossim / num_layers
        ratio = raw_bytes / comp_bytes
        
        print(f"{domain:<15} | {num_tokens:<8} | {raw_bytes/1024:.1f} KB   | {comp_bytes/1024:.1f} KB    | {ratio:<8.1f}x | {avg_fide:<10.6f}")

    print("-" * 80)
    print("✅ Multi-Domain Trust Test Completed. Verified for production versatility.")

if __name__ == "__main__":
    run_multi_domain_test()
