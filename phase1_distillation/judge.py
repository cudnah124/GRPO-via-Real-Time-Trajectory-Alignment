import torch
import itertools
from phase1_distillation.prompts import JUDGE_PROMPT, RETRY_PROMPT
from phase1_distillation.parser import parse_distance_matrix
from phase1_distillation.config import MAX_RETRIES

class AlignmentJudge:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, messages, max_new_tokens=1024):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2, # Low temp for deterministic evaluation
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
        input_length = inputs['input_ids'].shape[1]
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    def evaluate_single_pair(self, problem, rollout_a, rollout_b):
        """Evaluate a single pair with retry logic"""
        content = f"Problem: {problem}\n\nRollout A:\n{rollout_a}\n\nRollout B:\n{rollout_b}"
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": content}
        ]
        
        for attempt in range(MAX_RETRIES):
            try:
                raw_output = self._generate(messages)
                matrix = parse_distance_matrix(raw_output)
                return matrix
            except ValueError as e:
                # Nếu lỗi, truyền tin nhắn báo lỗi vào tiếp (nếu còn số lần retry)
                if attempt < MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": RETRY_PROMPT + f"\nError details: {e}"})
                else:
                    print(f"    [!] Failed after {MAX_RETRIES} attempts. Error: {e}")
                    
        return None

    def evaluate_pairs(self, problem, rollouts):
        """Evaluate all combinations C(K,2) of rollouts"""
        results = {}
        # Tạo tất cả các cặp (i, j) với i < j
        indices = list(range(len(rollouts)))
        pairs = list(itertools.combinations(indices, 2))
        
        for i, j in pairs:
            print(f"    [*] Evaluating pair ({i},{j})...")
            matrix = self.evaluate_single_pair(problem, rollouts[i], rollouts[j])
            results[f"({i},{j})"] = matrix
            
        return results
