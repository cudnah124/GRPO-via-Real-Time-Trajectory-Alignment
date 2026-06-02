import json
import os
import numpy as np

def analyze_file(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return
        
    costs = []
    problem_count = 0
    rollout_count = 0
    pair_count = 0
    zero_count = 0
    one_count = 0
    other_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            problem_count += 1
            rollout_count += len(item.get('generated_rollouts', []))
            
            costs_dict = item.get('alignment_costs', {})
            for key, cost_val in costs_dict.items():
                cost = float(cost_val)
                costs.append(cost)
                pair_count += 1
                if cost == 0.0:
                    zero_count += 1
                elif cost == 1.0:
                    one_count += 1
                else:
                    other_count += 1

    print("\n" + "="*60)
    print(f"THỐNG KÊ TẬP DỮ LIỆU: {jsonl_path.upper()}")
    print("="*60)
    print(f"Tổng số bài toán (problems):        {problem_count}")
    print(f"Tổng số lời giải (rollouts):        {rollout_count}")
    print(f"Tổng số cặp đối sánh (pairs):       {pair_count}")
    print(f"  + Cặp khớp hoàn toàn (cost = 0.0): {zero_count} ({zero_count/pair_count*100:.2f}%)")
    print(f"  + Cặp sai lệch hoàn toàn (cost = 1.0): {one_count} ({one_count/pair_count*100:.2f}%)")
    print(f"  + Cặp sai lệch một phần (0.0 < cost < 1.0): {other_count} ({other_count/pair_count*100:.2f}%)")
    
    # Histogram phân bổ
    if costs:
        hist, bin_edges = np.histogram(costs, bins=10, range=(0, 1))
        print("\nChi tiết phân bổ khoảng cách (10 khoảng từ 0.0 đến 1.0):")
        for i in range(10):
            print(f"  [{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}]: {hist[i]} ({hist[i]/pair_count*100:.2f}%)")
    print("="*60)

def main():
    # Phân tích cả file gốc và file tăng cường nếu tồn tại
    if os.path.exists("distillation_data.jsonl"):
        analyze_file("distillation_data.jsonl")
    if os.path.exists("distillation_data_augmented.jsonl"):
        analyze_file("distillation_data_augmented.jsonl")

if __name__ == "__main__":
    main()
