import json
import os

def compute_dtw_with_path(matrix):
    N, M = len(matrix), len(matrix[0])
    dp = [[float('inf')] * (M + 1) for _ in range(N + 1)]
    parent = [[None] * (M + 1) for _ in range(N + 1)]
    
    dp[0][0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = float(matrix[i-1][j-1])
            
            # Các hướng di chuyển đến (i, j)
            # 1. Từ (i-1, j-1) - Chéo
            # 2. Từ (i-1, j) - Dọc (xuống)
            # 3. Từ (i, j-1) - Ngang (sang phải)
            choices = [
                (dp[i-1][j-1], (i-1, j-1), "diag"),
                (dp[i-1][j], (i-1, j), "down"),
                (dp[i][j-1], (i, j-1), "right")
            ]
            
            best_val, best_prev, move_type = min(choices, key=lambda x: x[0])
            dp[i][j] = cost + best_val
            parent[i][j] = (best_prev, move_type)
            
    # Truy vết đường đi ngược từ (N, M) về (0, 0)
    path = []
    curr = (N, M)
    while curr != (0, 0) and curr is not None:
        p_info = parent[curr[0]][curr[1]]
        if p_info is not None:
            prev, move_type = p_info
            path.append((curr[0] - 1, curr[1] - 1, move_type, matrix[curr[0] - 1][curr[1] - 1]))
            curr = prev
        else:
            break
            
    path.reverse()
    raw_score = dp[N][M]
    norm_score = raw_score / (N + M - 1)
    return norm_score, raw_score, path

def main():
    file_path = "data/judge/01ac749b83158c27cf916ac852eb5a0d.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("=== KIỂM TRA ĐỐI SÁNH DTW TRÊN DỮ LIỆU THỰC TẾ ===")
    
    # 1. Xét cặp (0, 2): Rollout 0 (dài 13 bước) vs Rollout 2 (dài 7 bước)
    matrix_02 = data["distance_matrices"]["(0,2)"]
    norm_02, raw_02, path_02 = compute_dtw_with_path(matrix_02)
    
    print("\n[CẶP 0 vs 2] (Quy trình biến đổi đại số tương đồng nhưng độ dài khác biệt 13 vs 7 bước)")
    print(f"-> Điểm DTW Chuẩn hóa: {norm_02:.4f} (Raw: {raw_02:.2f})")
    print("Đường đi căn chỉnh tối ưu (Warping Path):")
    for step in path_02:
        print(f"   - Bước Rollout0[{step[0]}] khớp với Rollout2[{step[1]}] ({step[2]:4s}) | Chi phí ô: {step[3]}")

    # 2. Xét cặp (1, 3): Rollout 1 (16 bước) vs Rollout 3 (5 bước) - Rollout 3 là một lời giải bị cụt
    # Hãy xem DTW phát hiện lời giải bị cụt ở Rollout 3 như thế nào
    matrix_13 = data["distance_matrices"]["(1,3)"]
    norm_13, raw_13, path_13 = compute_dtw_with_path(matrix_13)
    
    print("\n[CẶP 1 vs 3] (Rollout 1 hoàn chỉnh 16 bước vs Rollout 3 bị cụt chỉ có 5 bước)")
    print(f"-> Điểm DTW Chuẩn hóa: {norm_13:.4f} (Raw: {raw_13:.2f})")
    print("Đường đi căn chỉnh tối ưu (Warping Path):")
    for step in path_13:
        print(f"   - Bước Rollout1[{step[0]}] khớp với Rollout3[{step[1]}] ({step[2]:4s}) | Chi phí ô: {step[3]}")

if __name__ == "__main__":
    main()
