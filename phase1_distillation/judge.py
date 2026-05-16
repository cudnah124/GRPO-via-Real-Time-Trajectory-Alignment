import os
import json
import itertools
from phase1_distillation.prompts import JUDGE_PROMPT
from phase1_distillation.parser import parse_distance_matrix
import phase1_distillation.config as config

class AlignmentJudge:
    def __init__(self, model_id=config.JUDGE_MODEL_ID):
        print(f"[*] Initializing LOCAL vLLM Judge with: {model_id}...")
        self.model_id = model_id
        
        # Lazy import vLLM
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it to use the local judge.")
        
        # Khởi tạo engine vLLM cho Judge
        self.llm = LLM(
            model=model_id,
            quantization="awq", # Sử dụng AWQ cho bản chính chủ Qwen
            gpu_memory_utilization=0.9, 
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=2048
        )

    def evaluate_batch(self, problems_data, cache_dir=None):
        """
        Evaluate multiple problems in parallel.
        problems_data: list of dicts {"id": ..., "problem": ..., "rollouts": ...}
        """
        all_prompts = []
        mapping = [] # Để biết prompt nào thuộc về bài nào, cặp nào
        results_by_prob = {d['id']: {} for d in problems_data}
        
        # 1. Thu thập toàn bộ các cặp từ tất cả các bài toán
        for data in problems_data:
            p_id, prob, rollouts = data['id'], data['problem'], data['rollouts']
            if len(rollouts) < 2: continue
            
            # Kiểm tra cache trước
            cache_file = os.path.join(cache_dir, f"judge_{p_id}.json") if cache_dir else None
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        results_by_prob[p_id] = json.load(f)
                except: pass

            indices = list(range(len(rollouts)))
            for i, j in itertools.combinations(indices, 2):
                pair_key = f"({i},{j})"
                if pair_key not in results_by_prob[p_id] or results_by_prob[p_id][pair_key] is None:
                    # SMART TRUNCATION LOGIC
                    tokenizer = self.llm.get_tokenizer()
                    
                    # Tính toán budget
                    sys_prompt = f"<|im_start|>system\n{JUDGE_PROMPT}<|im_end|>\n<|im_start|>user\nProblem: {prob}\n\nRollout A:\n"
                    mid_prompt = "\n\nRollout B:\n"
                    end_prompt = "<|im_end|>\n<|im_start|>assistant\n"
                    
                    base_len = len(tokenizer.encode(sys_prompt + mid_prompt + end_prompt))
                    budget_per_rollout = (4096 - base_len - 128) // 2 # 128 tokens dự phòng cho output
                    
                    # Tokenize và cắt tỉa
                    t_a = tokenizer.encode(rollouts[i])
                    t_b = tokenizer.encode(rollouts[j])
                    
                    if len(t_a) > budget_per_rollout:
                        t_a = t_a[:budget_per_rollout]
                    if len(t_b) > budget_per_rollout:
                        t_b = t_b[:budget_per_rollout]
                        
                    clean_a = tokenizer.decode(t_a)
                    clean_b = tokenizer.decode(t_b)
                    
                    # Reconstruct prompt
                    content = f"Problem: {prob}\n\nRollout A:\n{clean_a}\n\nRollout B:\n{clean_b}"
                    prompt = f"<|im_start|>system\n{JUDGE_PROMPT}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
                    
                    all_prompts.append(prompt)
                    mapping.append((p_id, pair_key))

        # 2. Chạy vLLM một lần cho TẤT CẢ các cặp
        if all_prompts:
            print(f"    [*] vLLM Parallel Judging {len(all_prompts)} pairs across {len(problems_data)} problems...")
            outputs = self.llm.generate(all_prompts, self.sampling_params)
            
            for (p_id, pair_key), output in zip(mapping, outputs):
                raw_text = output.outputs[0].text
                try:
                    matrix = parse_distance_matrix(raw_text)
                    results_by_prob[p_id][pair_key] = matrix
                except Exception as e:
                    results_by_prob[p_id][pair_key] = None

            # 3. Lưu cache cho từng bài
            if cache_dir:
                for p_id in results_by_prob:
                    if results_by_prob[p_id]:
                        c_file = os.path.join(cache_dir, f"judge_{p_id}.json")
                        with open(c_file, "w", encoding="utf-8") as f:
                            json.dump(results_by_prob[p_id], f, ensure_ascii=False, indent=2)

        return results_by_prob

    def evaluate_pairs(self, problem, problem_id, rollouts, cache_dir=None):
        """Giữ lại hàm cũ để tương thích ngược, nhưng gọi sang evaluate_batch"""
        data = {"id": problem_id, "problem": problem, "rollouts": rollouts}
        batch_results = self.evaluate_batch([data], cache_dir=cache_dir)
        return batch_results.get(problem_id, {})
