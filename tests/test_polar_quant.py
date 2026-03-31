import pytest
import torch
import math
from turboquant.core.polar_quant import PolarQuantizer, _pack_3bit, _unpack_3bit

class TestPolarQuantizerV2:
    def test_true_3bit_memory(self):
        head_dim = 128
        bits = 3
        q = PolarQuantizer(head_dim=head_dim, bits=bits)
        x = torch.randn(1, 1, 1, head_dim)
        packed, scales = q.forward(x)
        
        # Expected packed bytes for head_dim=128: 128 * 3 / 8 = 48 bytes
        expected_bytes = math.ceil(head_dim * bits / 8)
        assert packed.shape[-1] == expected_bytes
        
    def test_pack_unpack_roundtrip(self):
        # Test the bit packing logic independently
        n = 64 # mult of 8
        indices = torch.randint(0, 8, (1, n), dtype=torch.uint8)
        packed = _pack_3bit(indices)
        unpacked = _unpack_3bit(packed, n)
        assert torch.all(indices == unpacked)
        
    def test_rotation_orthogonality(self):
        head_dim = 64
        q = PolarQuantizer(head_dim=head_dim)
        pi = q.Pi
        # Q @ Q.T should be I
        eye = torch.eye(head_dim, device=pi.device)
        assert torch.allclose(pi @ pi.T, eye, atol=1e-4)
        
    def test_inverse_rotation_correctness(self):
        head_dim = 64
        q = PolarQuantizer(head_dim=head_dim, use_hadamard=False)
        x = torch.randn(1, 1, 1, head_dim)
        x_rot = q._apply_rotation(x, inverse=False)
        x_rec = q._apply_rotation(x_rot, inverse=True)
        assert torch.allclose(x, x_rec, atol=1e-5)
        
    def test_hadamard_vs_matmul_equivalence(self):
        head_dim = 64
        q_had = PolarQuantizer(head_dim=head_dim, use_hadamard=True)
        q_mat = PolarQuantizer(head_dim=head_dim, use_hadamard=False)
        
        x = torch.randn(1, 1, 1, head_dim)
        # They shouldn't be equal (different orthogonal transforms), but both should be orthonormal
        y_had = q_had._apply_rotation(x, inverse=False)
        y_mat = q_mat._apply_rotation(x, inverse=False)
        
        assert torch.allclose(x.norm(), y_had.norm(), atol=1e-4)
        assert torch.allclose(x.norm(), y_mat.norm(), atol=1e-4)
        
    def test_streaming_calibration(self):
        head_dim = 64
        q = PolarQuantizer(head_dim=head_dim)
        x_list = [torch.randn(1, 1, 32, head_dim) for _ in range(3)]
        
        # Current state (initialized with 0.5, 0.5)
        lev_init = q.levels.clone()
        
        q.calibrate(x_list)
        lev_final = q.levels.clone()
        
        # Levels should change after calibration on normal data
        assert not torch.allclose(lev_init, lev_final)
        
    def test_memory_footprint(self):
        head_dim = 128
        q = PolarQuantizer(head_dim=head_dim)
        footprint = q.memory_footprint_bytes(seq_len=1024, batch=1, heads=32)
        
        # FP16 size = 1 * 32 * 1024 * 128 * 2 = 8,388,608 bytes
        fp16_bytes = 1 * 32 * 1024 * head_dim * 2
        # Current footprint: (48 + (128/64)*4) * 32 * 1024 = (48 + 8) * 32768 = 1,835,008 bytes
        # Ratio = 1,835,008 / 8,388,608 ≈ 0.218 (< 0.25)
        assert footprint < fp16_bytes / 4
        
    def test_edge_cases(self):
        # head_dim = 1
        q1 = PolarQuantizer(head_dim=1)
        x1 = torch.randn(1, 1, 1, 1)
        p1, s1 = q1.forward(x1)
        assert p1.shape[-1] == 3
        
        # head_dim = 65 (not power of 2) -> padded to 72 indices -> 27 bytes
        q65 = PolarQuantizer(head_dim=65, use_hadamard=True)
        x65 = torch.randn(1, 1, 1, 65)
        p65, s65 = q65.forward(x65)
        assert p65.shape[-1] == 27
        
        x_rec = q65.dequantize(p65, s65)
        assert x_rec.shape == x65.shape
