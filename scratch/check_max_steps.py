import json
import os
import re
from collections import Counter

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

def main():
    jsonl_path = "distillation_data.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"[!] Lỗi: Không tìm thấy file '{jsonl_path}'")
        return
        
    print(f"[*] Đang quét tập dữ liệu '{jsonl_path}' để thống kê số lượng bước...")
    
    step_counts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rollouts = item.get("generated_rollouts", [])
            for rollout in rollouts:
                steps = get_steps_list(rollout)
                step_counts.append(len(steps))
                
    if not step_counts:
        print("[!] Không tìm thấy lời giải nào để thống kê.")
        return
        
    max_steps = max(step_counts)
    min_steps = min(step_counts)
    avg_steps = sum(step_counts) / len(step_counts)
    
    print("\n" + "="*60)
    print("THỐNG KÊ SỐ BƯỚC LẬP LUẬN TRONG TẬP DỮ LIỆU")
    print("="*60)
    print(f"Tổng số lời giải đã quét: {len(step_counts)}")
    print(f"Số bước ít nhất:           {min_steps}")
    print(f"Số bước nhiều nhất (Max):  {max_steps}")
    print(f"Số bước trung bình:        {avg_steps:.2f}")
    
    # Chi tiết phân bổ số lượng bước
    counter = Counter(step_counts)
    print("\nChi tiết phân bổ số lượng bước (Số bước: Tần suất xuất hiện):")
    for count in sorted(counter.keys()):
        freq = counter[count]
        pct = (freq / len(step_counts)) * 100
        print(f"  Lời giải có {count:2d} bước: {freq:5d} lần ({pct:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
