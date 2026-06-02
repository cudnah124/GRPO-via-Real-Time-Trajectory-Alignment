import os
import json
import re
import itertools

def get_steps_list(text):
    parts = re.split(r'(?i)(Step\s+\d+:)', text)
    steps = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        steps.append(f"{header} {content}")
    if not steps and text.strip():
        steps = [s.strip() for s in text.split("\n\n") if s.strip()]
    return steps

def verify_all_dimensions():
    judge_dir = os.path.join("data", "judge")
    if not os.path.exists(judge_dir):
        print(f"[!] Thư mục kết quả không tồn tại: {judge_dir}")
        return

    json_files = [f for f in os.listdir(judge_dir) if f.endswith(".json")]
    print(f"[*] Đang quét {len(json_files)} file kết quả trong {judge_dir}...")

    mismatches = []
    empty_files = []
    total_pairs_checked = 0
    mismatched_pairs = 0

    for filename in json_files:
        file_path = os.path.join(judge_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Lỗi đọc file {filename}: {e}")
            continue

        rollouts = data.get("generated_rollouts", [])
        distance_matrices = data.get("distance_matrices", {})

        if not rollouts or not distance_matrices:
            empty_files.append(filename)
            continue

        num_rollouts = len(rollouts)
        indices = list(range(num_rollouts))

        for r_a, r_b in itertools.combinations(indices, 2):
            key = f"({r_a},{r_b})"
            matrix = distance_matrices.get(key, [])
            
            steps_a = get_steps_list(rollouts[r_a])
            steps_b = get_steps_list(rollouts[r_b])

            expected_n = len(steps_a)
            expected_m = len(steps_b)

            if not matrix:
                # Ma trận trống (do lỗi API chưa chấm)
                mismatches.append({
                    "filename": filename,
                    "pair": key,
                    "expected": f"{expected_n}x{expected_m}",
                    "actual": "Empty/None",
                    "reason": "Chưa được đánh giá (Empty matrix)"
                })
                mismatched_pairs += 1
                total_pairs_checked += 1
                continue

            actual_n = len(matrix)
            actual_m = len(matrix[0]) if actual_n > 0 else 0

            total_pairs_checked += 1

            if actual_n != expected_n or actual_m != expected_m:
                mismatches.append({
                    "filename": filename,
                    "pair": key,
                    "expected": f"{expected_n}x{expected_m}",
                    "actual": f"{actual_n}x{actual_m}",
                    "reason": "Sai kích thước ma trận so với số bước"
                })
                mismatched_pairs += 1

    print("\n" + "="*60)
    print("BÁO CÁO KIỂM TRA KÍCH THƯỚC MA TRẬN KẾT QUẢ")
    print("="*60)
    print(f"[*] Tổng số file đã quét: {len(json_files)}")
    print(f"[*] Tổng số cặp ma trận đã kiểm tra: {total_pairs_checked}")
    print(f"[+] Số cặp khớp hoàn toàn kích thước: {total_pairs_checked - mismatched_pairs}")
    print(f"[!] Số cặp bị sai lệch kích thước / trống: {mismatched_pairs}")

    if mismatches:
        print("\n[!] DANH SÁCH CÁC CẶP BỊ LỖI KÍCH THƯỚC HOẶC RỖNG:")
        # Gom nhóm theo filename để hiển thị gọn
        from collections import defaultdict
        file_to_errors = defaultdict(list)
        for err in mismatches:
            file_to_errors[err["filename"]].append(err)

        for fname, errs in file_to_errors.items():
            print(f"\n- File: {fname}")
            for err in errs:
                print(f"  + Cặp {err['pair']}: Dự kiến {err['expected']} | Thực tế {err['actual']} ({err['reason']})")
    else:
        print("\n[+] TUYỆT VỜI: Tất cả ma trận đều khớp hoàn hảo với số lượng bước thực tế!")
    print("="*60)

if __name__ == "__main__":
    verify_all_dimensions()
