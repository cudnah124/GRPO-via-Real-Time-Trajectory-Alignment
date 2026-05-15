import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Initializing vLLM engine with: {model_id}...")
        self.model_id = model_id
        
        # Khởi tạo engine vLLM
        # gpu_memory_utilization=0.8 để dành 20% VRAM cho các tác vụ khác
        # max_model_len giúp giới hạn chiều dài sequence để tiết kiệm KV cache
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            trust_remote_code=True
        )
        
        # Tokenizer chỉ dùng để format prompt template
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        """
        Sử dụng vLLM để sinh lời giải cực nhanh cho cả danh sách bài toán.
        """
        results = {}
        pending_problems = []
        pending_ids = []
        
        # 1. Lọc cache
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

        # 2. Chuẩn bị danh sách Prompts (dạng chuỗi văn bản)
        prompts = []
        for prob in pending_problems:
            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": prob}
            ]
            # vLLM generate trực tiếp từ text prompt
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt_text)

        # 3. Cấu hình tham số sinh (Sampling)
        # n=num_rollouts: vLLM sẽ sinh cùng lúc K kết quả cho MỖI prompt
        sampling_params = SamplingParams(
            n=num_rollouts, 
            temperature=0.7,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        print(f"    [*] vLLM Generating: {len(pending_problems)} problems (Total {len(pending_problems)*num_rollouts} sequences)...")
        
        try:
            # vLLM tự động Batching cực kỳ tối ưu
            outputs = self.llm.generate(prompts, sampling_params)
            
            # 4. Thu thập kết quả
            for i, output in enumerate(outputs):
                p_id = pending_ids[i]
                # Lấy danh sách văn bản từ list các Output của vLLM
                problem_rollouts = [out.text for out in output.outputs]
                
                results[p_id] = problem_rollouts
                
                # Lưu vào cache
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