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
            quantization="compressed-tensors", # Khớp với config.json của model
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

    def evaluate_pairs(self, problem, problem_id, rollouts, cache_dir=None):
        """Evaluate all combinations C(K,2) using LOCAL vLLM with batching"""
        if len(rollouts) < 2:
            return {}
            
        results = {}
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"judge_{problem_id}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    print(f"    [+] Loaded {len(results)} pairs from judge cache.")
                except:
                    results = {}

        indices = list(range(len(rollouts)))
        all_pairs = list(itertools.combinations(indices, 2))
        
        pending_pairs = []
        prompts = []
        for i, j in all_pairs:
            pair_key = f"({i},{j})"
            if pair_key not in results or results[pair_key] is None:
                pending_pairs.append(pair_key)
                content = f"Problem: {problem}\n\nRollout A:\n{rollouts[i]}\n\nRollout B:\n{rollouts[j]}"
                prompt = f"<|im_start|>system\n{JUDGE_PROMPT}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(prompt)

        if prompts:
            print(f"    [*] Batch Judging {len(prompts)} pairs locally...")
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            for pair_key, output in zip(pending_pairs, outputs):
                raw_text = output.outputs[0].text
                try:
                    matrix = parse_distance_matrix(raw_text)
                    results[pair_key] = matrix
                except Exception as e:
                    print(f"    [!] Failed to parse matrix for {pair_key}: {e}")
                    results[pair_key] = None

            if cache_file:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results
