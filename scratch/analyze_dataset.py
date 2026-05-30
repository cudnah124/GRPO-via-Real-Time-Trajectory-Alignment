import json
import os
import matplotlib.pyplot as plt
import numpy as np

def compute_teacher_dtw(matrix):
    N, M = len(matrix), len(matrix[0])
    dp = [[float('inf')] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = float(matrix[i-1][j-1])
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[N][M] / max(N, M)

def main():
    jsonl_path = r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\distillation_data_augmented.jsonl"
    if not os.path.exists(jsonl_path):
        print("File not found:", jsonl_path)
        return
    
    costs = []
    pair_count = 0
    zero_count = 0
    one_count = 0
    other_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            item = json.loads(line)
            matrices = item.get('distance_matrices', {})
            for key, mat in matrices.items():
                if not (mat and isinstance(mat, list) and isinstance(mat[0], list)):
                    continue
                cost = compute_teacher_dtw(mat)
                costs.append(cost)
                pair_count += 1
                if cost == 0.0:
                    zero_count += 1
                elif cost == 1.0:
                    one_count += 1
                else:
                    other_count += 1

    print("\n--- DATASET STATISTICS ---")
    print(f"Total pairs analyzed: {pair_count}")
    print(f"Pairs with cost = 0.0: {zero_count} ({zero_count/pair_count*100:.2f}%)")
    print(f"Pairs with cost = 1.0: {one_count} ({one_count/pair_count*100:.2f}%)")
    print(f"Pairs with 0.0 < cost < 1.0: {other_count} ({other_count/pair_count*100:.2f}%)")
    
    # Histogram distribution
    hist, bin_edges = np.histogram(costs, bins=10, range=(0, 1))
    print("\nCost distribution (10 bins):")
    for i in range(10):
        print(f"  [{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}]: {hist[i]} ({hist[i]/pair_count*100:.2f}%)")

if __name__ == "__main__":
    main()
