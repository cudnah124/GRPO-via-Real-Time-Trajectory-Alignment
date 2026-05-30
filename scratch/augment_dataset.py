import json
import os
import re
import random
from collections import defaultdict
from tqdm import tqdm

mutation_log = defaultdict(lambda: {"success": 0, "fail": 0})

def mutate_sign(text):
    def invert_signs(match):
        eq = match.group(0)
        # Đảo ngược dấu và các toán tử so sánh (để tăng tỷ lệ thành công)
        eq = eq.replace('=', '\\neq ').replace('<', '>').replace('>', '<')
        trans = str.maketrans({'+': '-', '-': '+'})
        return eq.translate(trans)
    return re.sub(r'(\$.*?\$|\\\(.*?\\\))', invert_signs, text)

def mutate_keyword(text):
    swaps = {
        r'\blinearly dependent\b': 'linearly independent',
        r'\blinearly independent\b': 'linearly dependent',
        r'\bminimum\b': 'maximum',
        r'\bmaximum\b': 'minimum',
        r'\bconverge\b': 'diverge',
        r'\bdiverge\b': 'converge',
        r'\bcyclic\b': 'non-cyclic',
        r'\bnon-cyclic\b': 'cyclic',
        r'\bequal to\b': 'not equal to',
        r'\blie on\b': 'do not lie on',
        r'\beven\b': 'odd',
        r'\bodd\b': 'even',
        r'\bpositive\b': 'negative',
        r'\bnegative\b': 'positive',
        r'\btrue\b': 'false',
        r'\bfalse\b': 'true',
        r'\bincreasing\b': 'decreasing',
        r'\bdecreasing\b': 'increasing',
        r'\bparallel\b': 'perpendicular',
        r'\bperpendicular\b': 'parallel'
    }
    mutated = text
    for pattern, replacement in swaps.items():
        if re.search(pattern, mutated, flags=re.IGNORECASE):
            mutated = re.sub(pattern, replacement, mutated, flags=re.IGNORECASE)
            break
    return mutated

def mutate_negation(text):
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    if len(sentences) > 0:
        first = sentences[0]
        if " is " in first and " is not " not in first:
            sentences[0] = first.replace(" is ", " is not ", 1)
        elif " are " in first and " are not " not in first:
            sentences[0] = first.replace(" are ", " are not ", 1)
        else:
            sentences[0] = "Assume the contrary that " + first
        return " ".join(sentences)
    return text

def mutate_quantifier(text):
    mutated = text
    if r"\forall" in mutated or r"\exists" in mutated:
        mutated = mutated.replace(r"\forall", r"\exists").replace(r"\exists", r"\forall")
    elif "for all" in mutated or "for some" in mutated:
        mutated = mutated.replace("for all", "for some").replace("for some", "for all")
    elif "for any" in mutated:
        mutated = mutated.replace("for any", "for some")
    return mutated

def mutate_deletion(text):
    sentences = re.split(r'(?<=\.)\s+', text)
    if len(sentences) >= 4:
        idx_to_delete = random.choice(range(1, len(sentences) - 1))
        sentences.pop(idx_to_delete)
        return " ".join(sentences)
    return text

def mutate_scrambling(text):
    sentences = re.split(r'(?<=\.)\s+', text)
    if len(sentences) >= 4:
        mid = len(sentences) // 2
        return " ".join(sentences[mid:]) + " " + " ".join(sentences[:mid])
    return text

def mutate_notation(text):
    swaps = {
        r'\\leq': r'\\geq',
        r'\\geq': r'\\leq',
        r'\\subset': r'\\supset',
        r'\\supset': r'\\subset',
        r'\\cup': r'\\cap',
        r'\\cap': r'\\cup',
        r'P\(A\|B\)': r'P(A/B)',
        r'f\'\(x\)': r'f/(x)'
    }
    mutated = text
    for pattern, replacement in swaps.items():
        if re.search(pattern, mutated):
            mutated = re.sub(pattern, replacement, mutated)
            break
    return mutated

def mutate_shortcut(text):
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    if len(sentences) >= 4:
        return sentences[0] + " " + sentences[-1]
    return text

def mutate_number(text):
    numbers = list(set(re.findall(r'\b\d+\b', text)))
    if not numbers:
        return text
    target = random.choice(numbers)
    new_val = str(int(target) + 1)
    return re.sub(rf'\b{target}\b', new_val, text, count=1)

def mutate_circular(text):
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    if len(sentences) >= 3:
        last_sent = sentences[-1]
        last_sent = re.sub(r'^(Therefore|Thus|Hence|So|We conclude that)[,\s]*', 'We know that ', last_sent, flags=re.IGNORECASE)
        return last_sent + " " + text
    return text

def mutate_proof_direction(text):
    pattern = r"(?i)if\s+(.+?)[,\s]+then\s+(.+?)(?=[.\n]|$)"
    match = re.search(pattern, text)
    if match:
        A = match.group(1)
        B = match.group(2)
        # Sử dụng lambda để re.sub không parse các ký tự backslash (\c, \f) trong LaTeX như escape sequence
        return re.sub(pattern, lambda m: f"If {B}, then {A}", text, count=1)
    return text

def mutate_conclusion_swap(text):
    fake_conclusions = [
        " Therefore, the answer is $0$.",
        " Thus, the sequence diverges.",
        " Hence, there are no real solutions.",
        " So the vectors are linearly dependent.",
        " We conclude that $P = 1/2$.",
        " The maximum value is 0."
    ]
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    if len(sentences) >= 3:
        sentences[-1] = random.choice(fake_conclusions)
        return " ".join(sentences)
    return text

def mutate_noise(text):
    noise_steps = [
        " Recall that the derivative of e^x is e^x. ",
        " Note that the probability of a coin flip is 0.5. ",
        " Using the Pythagorean theorem, we get 5. "
    ]
    if len(text) < 20:
        return text
    mid = len(text) // 2
    match = re.search(r'\s+', text[mid:])
    if match:
        idx = mid + match.start()
        mutated = text[:idx] + random.choice(noise_steps) + text[idx:]
        if len(mutated) > len(text):
            return mutated
    return text

def mutate_fatal_logic(text):
    """Quy tắc 0.5 (Mới): Lỗi logic chí mạng (Universal Fallback)"""
    sentences = re.split(r'(?<=\.)\s+', text)
    if len(sentences) >= 3:
        # Chèn mệnh đề sai hoàn toàn vào đầu
        sentences.insert(1, "However, since $1 = 0$, all conditions are trivially satisfied.")
        return " ".join(sentences)
    return text

# Nhóm các mutators theo Severity
RULES_05 = [mutate_sign, mutate_quantifier, mutate_shortcut, mutate_keyword, mutate_negation, mutate_fatal_logic]
RULES_04 = [mutate_deletion, mutate_proof_direction, mutate_notation, mutate_scrambling]
RULES_03 = [mutate_noise, mutate_number, mutate_circular, mutate_conclusion_swap]

def apply_with_log(rule_func, rollout):
    result = rule_func(rollout)
    rule_name = rule_func.__name__
    if result != rollout:
        mutation_log[rule_name]["success"] += 1
        return result
    else:
        mutation_log[rule_name]["fail"] += 1
        return None

def apply_stratified_mutation(base_rollout, target_severity):
    if target_severity == 0.5:
        rules = list(RULES_05)
    elif target_severity == 0.4:
        rules = list(RULES_04)
    else:
        rules = list(RULES_03)
        
    random.shuffle(rules)
    for rule in rules:
        mutated = apply_with_log(rule, base_rollout)
        if mutated is not None:
            return mutated, rule.__name__
            
    # Fallback: Nếu mọi rule trong severity này đều fail, có thể ghi log
    return None, None

def main():
    input_path = r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\distillation_data.jsonl"
    output_path = r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\distillation_data_augmented.jsonl"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    print("[*] Loading original dataset and applying Stratified Augmentation...")
    
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    augmented_count = 0
    severity_cycle = [0.3, 0.4, 0.5]
    cycle_idx = 0

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Augmenting Dataset"):
            item = json.loads(line)
            rollouts = item.get('generated_rollouts', [])
            matrices = item.get('distance_matrices', {})
            
            if not rollouts or not isinstance(rollouts, list):
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue
                
            # Tạo 2 Negatives cho mỗi bài toán để đạt ~48,000 cặp Negatives tổng cộng (Tỷ lệ Pos:Neg ~ 1:1.3)
            if len(rollouts) > 0:
                for _ in range(2):
                    idx_to_mutate = random.randint(0, len(rollouts) - 1)
                    base_rollout = rollouts[idx_to_mutate]
                    
                    # Xoay vòng target severity: 0.3 -> 0.4 -> 0.5
                    target_severity = severity_cycle[cycle_idx % 3]
                    cycle_idx += 1
                    
                    mutated_rollout, mutator_name = apply_stratified_mutation(base_rollout, target_severity)
                    
                    if mutated_rollout:
                        mutated_idx = len(rollouts)
                        rollouts.append(mutated_rollout)
                        
                        base_steps_count = len(re.findall(r'(?i)(step\s+\d+[:\.]|bước\s+\d+[:\.])', base_rollout)) or 3
                        mutated_steps_count = len(re.findall(r'(?i)(step\s+\d+[:\.]|bước\s+\d+[:\.])', mutated_rollout)) or 3
                        
                        # Gán nhãn cứng cho base_rollout vs mutated_rollout
                        bad_matrix = [[target_severity] * mutated_steps_count for _ in range(base_steps_count)]
                        pair_key = f"({idx_to_mutate},{mutated_idx})"
                        matrices[pair_key] = bad_matrix
                        
                        # Cập nhật nhãn cứng giữa mutated_rollout và các rollout đúng khác
                        for other_idx in range(len(rollouts) - 1):
                            if other_idx != idx_to_mutate:
                                other_steps_count = len(re.findall(r'(?i)(step\s+\d+[:\.]|bước\s+\d+[:\.])', rollouts[other_idx])) or 3
                                m_key = f"({min(other_idx, mutated_idx)},{max(other_idx, mutated_idx)})"
                                matrices[m_key] = [[target_severity] * mutated_steps_count for _ in range(other_steps_count)]
                        
                        augmented_count += 1
                    
            item['generated_rollouts'] = rollouts
            item['distance_matrices'] = matrices
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[+] DATA AUGMENTATION SUCCESSFUL!")
    print(f"  - Original file: {input_path}")
    print(f"  - Augmented file: {output_path}")
    print(f"  - Total new negatives created: {augmented_count}")
    
    print("\n=== MUTATION SUCCESS LOG ===")
    for rule, stats in sorted(mutation_log.items(), key=lambda x: x[1]['success'], reverse=True):
        print(f"{rule:25s}: success={stats['success']:<5d}, fail={stats['fail']:<5d}")

if __name__ == "__main__":
    main()
