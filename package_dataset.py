import os
import json

def compute_teacher_dtw(matrix):
    # Trả về 1.0 ngay lập tức nếu tất cả các phần tử là 1.0 (cho tính nhất quán)
    is_cross = True
    for row in matrix:
        for val in row:
            if abs(float(val) - 1.0) > 1e-5:
                is_cross = False
                break
        if not is_cross:
            break
    if is_cross:
        return 1.0

    N, M = len(matrix), len(matrix[0])
    dp = [[float('inf')] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = float(matrix[i-1][j-1])
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[N][M] / (N + M - 1)

def package_dataset():
    judge_dir = os.path.join("data", "judge")
    output_file = "distillation_data.jsonl"
    
    if not os.path.exists(judge_dir):
        print(f"[!] Thư mục kết quả '{judge_dir}' không tồn tại.")
        return
        
    json_files = [f for f in os.listdir(judge_dir) if f.endswith(".json")]
    print(f"[*] Tìm thấy {len(json_files)} file kết quả trong thư mục '{judge_dir}'.")
    
    packed_count = 0
    skipped_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for fname in json_files:
            file_path = os.path.join(judge_dir, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    data = json.load(in_f)
                
                rollouts = data.get("generated_rollouts", [])
                matrices = data.get("distance_matrices", {})
                
                if not rollouts or not matrices:
                    print(f"  [!] Bỏ qua file trống hoặc thiếu dữ liệu: {fname}")
                    skipped_count += 1
                    continue
                
                # Tính toán DTW cho các cặp ma trận và lưu thành điểm số duy nhất
                alignment_costs = {}
                has_empty_matrix = False
                for key, mat in matrices.items():
                    if not mat or len(mat) == 0 or not isinstance(mat[0], list):
                        has_empty_matrix = True
                        break
                    alignment_costs[key] = round(compute_teacher_dtw(mat), 4)
                
                if has_empty_matrix:
                    print(f"  [!] Bỏ qua file chứa ma trận rỗng: {fname}")
                    skipped_count += 1
                    continue
                
                # Tạo bản ghi mới chỉ lưu rollouts và điểm số dtw
                packaged_item = {
                    "problem_id": data.get("problem_id", ""),
                    "generated_rollouts": rollouts,
                    "alignment_costs": alignment_costs
                }
                
                # Ghi một dòng JSONL duy nhất
                json_line = json.dumps(packaged_item, ensure_ascii=False)
                out_f.write(json_line + "\n")
                packed_count += 1
                
            except Exception as e:
                print(f"  [!] Lỗi khi xử lý file {fname}: {e}")
                skipped_count += 1
                
    print("\n" + "="*60)
    print("HOÀN TẤT ĐÓNG GÓI TẬP DỮ LIỆU HUẤN LUYỆN (ĐÃ TÍNH DTW SẴN)")
    print("="*60)
    print(f"[+] Tổng số bài toán đã đóng gói thành công: {packed_count}")
    print(f"[-] Số bài toán bị bỏ qua do lỗi/trống:       {skipped_count}")
    print(f"[+] Đường dẫn file dataset đầu ra:           {os.path.abspath(output_file)}")
    print("\n[*] File 'distillation_data.jsonl' giờ chỉ chứa điểm số DTW tĩnh.")
    print("    Không còn lưu trữ ma trận thô của Gemini, dung lượng giảm đáng kể!")
    print("="*60)

if __name__ == "__main__":
    package_dataset()
