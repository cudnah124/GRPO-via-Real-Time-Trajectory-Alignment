import os
import sys

# ✅ THIẾT LẬP MÔI TRƯỜNG TRƯỚC KHI IMPORT VLLM
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# Vá lỗi fileno cho cả tiến trình chính
import io
if not hasattr(sys.stdout, 'fileno'):
    try:
        sys.stdout.fileno = lambda: 1
    except:
        pass
if not hasattr(sys.stderr, 'fileno'):
    try:
        sys.stderr.fileno = lambda: 2
    except:
        pass

import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Initializing vLLM engine (V0) with: {model_id}...")
        self.model_id = model_id
        
        # Cấu hình engine vLLM V0
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=0.7, 
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True # Tắt log thống kê để tránh lỗi fileno khi in ấn
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        results = {}
        pending_problems = []
        pending_ids = []
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        for prob, p_id in zip(problems, problem_ids):
            cache_file = os.path.join(cache_dir, f"{p_id}.json") if cache_dir else None
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if len(cached) >= num_rollouts:
                        results[p_id] = cached[:num_rollouts]
                        continue
                except:
                    pass
            pending_problems.append(prob)
            pending_ids.append(p_id)
            results[p_id] = []

        if not pending_problems:
            return results

        prompts = []
        for prob in pending_problems:
            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": prob}
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt_text)

        sampling_params = SamplingParams(
            n=num_rollouts, 
            temperature=0.7,
            max_tokens=max_tokens,
        )

        print(f"    [*] vLLM (V0) Generating: {len(pending_problems)} problems...")
        
        try:
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)
            
            for i, output in enumerate(outputs):
                p_id = pending_ids[i]
                problem_rollouts = [out.text for out in output.outputs]
                results[p_id] = problem_rollouts
                
                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"{p_id}.json")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(problem_rollouts, f, ensure_ascii=False, indent=2)
                        
            print(f"    [+] vLLM Batch completed.")
            
        except Exception as e:
            print(f"    [!] vLLM Error: {e}")
            
        return results

    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens)
        return res.get(problem_id, [])