import sys
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

import os
import json
from transformers import AutoTokenizer
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            import traceback
            print(f"[!] vLLM import failed with error: {e}")
            traceback.print_exc()
            self.llm = None
            return

        print(f"[*] Initializing vLLM engine with: {model_id}...")
        self.model_id = model_id
        
        # Khởi tạo engine vLLM
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=0.7, 
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
        if self.llm is None:
            raise ImportError("vLLM is not installed. Cannot generate rollouts.")
        
        from vllm import SamplingParams
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

        print(f"    [*] vLLM Generating: {len(pending_problems)} problems...")
        
        try:
            # vLLM (V0) sẽ chạy mượt mà sau khi đã được Monkey Patch
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

    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens)
        return res.get(problem_id, [])