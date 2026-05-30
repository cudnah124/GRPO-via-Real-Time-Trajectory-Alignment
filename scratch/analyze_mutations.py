import json
import os
import re

# Định nghĩa các check để phân loại mutation
def detect_mutation_type(orig, mut):
    if orig == mut:
        return "none"
    
    # 1. Noise Step Insertion
    if "Step Extra:" in mut:
        return "mutate_noise (Noise step insertion)"
    
    # 2. Circular Reasoning
    if "Assume that the solution is correct." in mut:
        return "mutate_circular (Circular reasoning)"
    
    # 3. Step Deletion
    if "[Skipped justification step]" in mut:
        return "mutate_deletion (Justification step deletion)"
        
    # 4. Right Answer, Wrong Reasoning
    if "Let's guess some values for x" in mut:
        return "mutate_reasoning_trap (Right answer, wrong reasoning)"
        
    # 5. Premise Mutation
    if ("legs of length 5 and 12" in mut and "legs of length 3 and 4" in orig) or \
       ("at B" in mut and "at A" in orig) or ("at C" in mut and "at B" in orig):
        return "mutate_premise (Premise mutation)"

    # 6. Unit Mutation
    if "kilometers" in mut and "meters" in orig:
        return "mutate_unit (Unit/Dimension mutation)"

    # 7. Notation Trap
    if "P(A/B)" in mut or "f/(x)" in mut:
        return "mutate_notation (Notation trap)"

    # 8. Quantifier Mutation
    if ("\\exists" in mut and "\\forall" in orig) or ("\\forall" in mut and "\\exists" in orig):
        return "mutate_quantifier (Quantifier mutation)"

    # 9. Proof Direction
    if "necessary condition" in mut and "sufficient condition" in orig:
        return "mutate_proof_direction (Proof direction flip)"

    # 10. Keyword Swap (Ví dụ: dependent <-> independent)
    # So sánh xem có sự thay đổi từ khóa không
    keywords = ["dependent", "independent", "minimum", "maximum", "converge", "diverge", "cyclic", "non-cyclic", "equal to", "not equal to"]
    has_keyword_change = False
    for kw in keywords:
        if (kw in orig and kw not in mut) or (kw in mut and kw not in orig):
            has_keyword_change = True
            break
    if has_keyword_change:
        return "mutate_keyword (Logical keyword swap - Pair 7)"

    # 11. Sign Mutation (Đổi dấu toán học - Cặp 2, 9)
    # Nếu số lượng ký tự giống nhau nhưng các dấu + - bị đảo ngược
    orig_chars = re.sub(r'[\+\-\*/]', '', orig)
    mut_chars = re.sub(r'[\+\-\*/]', '', mut)
    if orig_chars == mut_chars:
        return "mutate_sign (Sign/Operator swap - Pair 2, 9)"

    # 12. Step Scrambling
    # Số lượng ký tự giống hệt nhau (sau khi xóa khoảng trắng/line break) nhưng thứ tự khác nhau
    orig_clean = "".join(orig.split())
    mut_clean = "".join(mut.split())
    if sorted(orig_clean) == sorted(mut_clean):
        return "mutate_scrambling (Step order scrambling)"

    # 13. Local Valid
    return "mutate_local_valid (Local valid, global invalid / Other)"

def main():
    jsonl_path = r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\distillation_data_augmented.jsonl"
    if not os.path.exists(jsonl_path):
        print("File not found:", jsonl_path)
        return
        
    mutation_stats = {}
    total_mutations = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            rollouts = item.get('generated_rollouts', [])
            
            # Trong code augment, ta chèn rollout đột biến vào cuối danh sách (index cuối cùng)
            # Và chỉ chèn tối đa 1 rollout đột biến mỗi bài.
            # Ta xác định xem bài toán này có bị đột biến không bằng cách kiểm tra
            # xem có cặp khóa nào trong distance_matrices trỏ đến index cuối cùng không
            matrices = item.get('distance_matrices', {})
            mutated_idx = len(rollouts) - 1
            
            has_mutation = False
            orig_idx = -1
            for key in matrices.keys():
                i, j = map(int, key.strip('()').split(','))
                if j == mutated_idx and i != j:
                    has_mutation = True
                    orig_idx = i
                    break
                    
            if has_mutation and orig_idx != -1:
                orig_text = rollouts[orig_idx]
                mut_text = rollouts[mutated_idx]
                
                m_type = detect_mutation_type(orig_text, mut_text)
                mutation_stats[m_type] = mutation_stats.get(m_type, 0) + 1
                total_mutations += 1

    print("\n=== DETAILED MUTATION METHOD STATISTICS ===")
    print(f"Total mutated samples generated: {total_mutations}\n")
    
    # Sắp xếp theo số lượng giảm dần
    sorted_stats = sorted(mutation_stats.items(), key=lambda x: x[1], reverse=True)
    for m_type, count in sorted_stats:
        percentage = (count / total_mutations) * 100
        print(f"  - {m_type}: {count} samples ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
