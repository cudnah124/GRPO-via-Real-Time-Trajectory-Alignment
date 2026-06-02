import os
import json

def check_progress():
    input_dir = os.path.join("data", "rollouts")
    output_dir = os.path.join("data", "judge")
    
    if not os.path.exists(input_dir):
        print(f"[!] Thư mục đầu vào không tồn tại: {input_dir}")
        return
        
    if not os.path.exists(output_dir):
        print(f"[*] Chưa có thư mục kết quả: {output_dir} (Đã xử lý: 0 file)")
        return

    # 1. Quét danh sách file
    input_files = set(f for f in os.listdir(input_dir) if f.endswith(".json"))
    output_files = set(f for f in os.listdir(output_dir) if f.endswith(".json"))
    
    total = len(input_files)
    completed = len(output_files.intersection(input_files))
    pending = len(input_files - output_files)
    
    # 2. Kiểm tra chất lượng các file đã chấm điểm
    corrupted_files = []
    empty_matrices_files = []
    
    for filename in output_files:
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Kiểm tra xem các ma trận khoảng cách có bị rỗng do lỗi API trước đó không
            matrices = data.get("distance_matrices", {})
            has_error = False
            for key, matrix in matrices.items():
                if not matrix or len(matrix) == 0:
                    has_error = True
                    break
            
            if has_error:
                empty_matrices_files.append(filename)
        except Exception:
            corrupted_files.append(filename)

    # 3. In báo cáo chi tiết
    print("="*60)
    print("BÁO CÁO TIẾN TRÌNH VÀ CHẤT LƯỢNG DỮ LIỆU JUDGE")
    print("="*60)
    print(f"[*] Tổng số bài toán đầu vào (data/rollout): {total}")
    print(f"[+] Số bài đã hoàn thành hoàn toàn:        {completed - len(empty_matrices_files)} / {total}")
    print(f"[-] Số bài chưa xử lý (Pending hoàn toàn):  {pending}")
    print(f"[!] Số bài đã xử lý nhưng bị thiếu ma trận: {len(empty_matrices_files)} (Sẽ được chấm bù tự động)")
    
    # 4. Tự động dọn dẹp các tệp bị lỗi nặng (hỏng JSON)
    errors_found = False
    
    if corrupted_files:
        errors_found = True
        print(f"\n[!] Phát hiện {len(corrupted_files)} file bị lỗi định dạng JSON (hỏng file). Đang tiến hành xóa để tự động chấm lại...")
        for filename in corrupted_files:
            file_path = os.path.join(output_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  [!] Không thể xóa {filename}: {e}")
                
    if empty_matrices_files:
        print(f"\n[*] GỢI Ý: Có {len(empty_matrices_files)} file bị thiếu một số ma trận (do lỗi API 429 trước đó).")
        print("    Bạn KHÔNG cần xóa chúng nữa. Chỉ cần chạy `python judge_demo.py`, script sẽ tự động chấm bù các cặp còn thiếu.")
        
    if errors_found:
        print("\n[*] ĐÃ DỌN DẸP XONG CÁC TỆP LỖI HỎNG JSON! Bạn chỉ cần chạy lại `python judge_demo.py` để bổ sung.")
    else:
        print("\n[+] Trạng thái: Sẵn sàng để chạy tiếp tục hoặc hoàn thành.")
    print("="*60)

if __name__ == "__main__":
    check_progress()
