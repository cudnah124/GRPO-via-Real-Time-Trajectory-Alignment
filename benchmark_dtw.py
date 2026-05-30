import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DiagonalSoftDTWSlow(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, D):
        B, N, M = D.shape
        device = D.device
        R = torch.full((B, N + 1, M + 1), 1e8, device=device)
        R[:, 0, 0] = 0
        
        for k in range(2, N + M + 1):
            i_start = max(1, k - M)
            i_end = min(N, k - 1)
            I = torch.arange(i_start, i_end + 1, device=device)
            J = k - I
            
            # Slice indexing: produces (L, B) shape
            v = torch.stack([
                R[:, I - 1, J],
                R[:, I, J - 1],
                R[:, I - 1, J - 1]
            ], dim=1)
            
            soft_min = -self.gamma * torch.logsumexp(-v / self.gamma, dim=1)
            R[:, I, J] = D[:, I - 1, J - 1] + soft_min
            
        return R[:, N, M]

class DiagonalSoftDTWFast(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, D):
        B, N, M = D.shape
        device = D.device
        R = torch.full((B, N + 1, M + 1), 1e8, device=device)
        R[:, 0, 0] = 0
        
        # Precompute batch_idx for advanced indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(-1) # (B, 1)
        
        for k in range(2, N + M + 1):
            i_start = max(1, k - M)
            i_end = min(N, k - 1)
            I = torch.arange(i_start, i_end + 1, device=device)
            J = k - I
            
            # Advanced indexing with batch_idx: produces (B, L) shape directly
            v = torch.stack([
                R[batch_idx, I - 1, J],
                R[batch_idx, I, J - 1],
                R[batch_idx, I - 1, J - 1]
            ], dim=1) # (B, 3, L)
            
            soft_min = -self.gamma * torch.logsumexp(-v / self.gamma, dim=1) # (B, L)
            R[batch_idx, I, J] = D[batch_idx, I - 1, J - 1] + soft_min
            
        return R[:, N, M]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    B, N, M = 128, 128, 128
    D = torch.rand(B, N, M, device=device)
    
    slow_dtw = DiagonalSoftDTWSlow().to(device)
    fast_dtw = DiagonalSoftDTWFast().to(device)
    
    # Warmup
    _ = slow_dtw(D)
    _ = fast_dtw(D)
    
    # Timing Slow
    t0 = time.time()
    for _ in range(20):
        res_slow = slow_dtw(D)
    t1 = time.time()
    print(f"Slow (Slice Indexing) time: {t1 - t0:.4f}s")
    
    # Timing Fast
    t0 = time.time()
    for _ in range(20):
        res_fast = fast_dtw(D)
    t1 = time.time()
    print(f"Fast (Batch Indexing) time: {t1 - t0:.4f}s")
    
    diff = torch.abs(res_slow - res_fast).max().item()
    print(f"Difference: {diff}")
