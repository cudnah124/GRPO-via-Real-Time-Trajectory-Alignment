import os
import sys
import json
import io

# ✅ VÁ LỖI Colab: Giả lập fileno cho sys.stdout/stderr
# vLLM cần cái này để khởi tạo engine trong môi trường Notebook
if not hasattr(sys.stdout, 'fileno'):
    sys.stdout.fileno = lambda: 1
if not hasattr(sys.stderr, 'fileno'):
    sys.stderr.fileno = lambda: 2

# Tắt vLLM V1 (đang thử nghiệm) để dùng bản V0 ổn định hơn trên Colab
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Initializing vLLM engine (V0) with: {model_id}...")
        self.model_id = model_id
        
        # Khởi tạo engine vLLM
        # Giảm gpu_memory_utilization xuống 0.7 để an toàn tuyệt đối trên T4
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=0.7, 
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True # Giúp tránh lỗi khởi tạo CUDA phức tạp
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

        # Chuẩn bị Prompts
        prompts = []
        for prob in pending_problems:
            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": prob}
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt_text)

        # Cấu hình Sampling
        sampling_params = SamplingParams(
            n=num_rollouts, 
            temperature=0.7,
            max_tokens=max_tokens,
        )

        print(f"    [*] vLLM Generating: {len(pending_problems)} problems...")
        
        try:
            outputs = self.llm.generate(prompts, sampling_params)
            
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