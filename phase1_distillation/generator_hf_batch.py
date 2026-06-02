import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Initializing batched Hugging Face model: {model_id}...")
        self.model_id = model_id
        
        # Cấu hình Tokenizer hỗ trợ Left-Padding cho Batch Generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or self.tokenizer.bos_token_id or 0

        # Load model ở Float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048, batch_size=16):
        results = {}
        pending_problems = []
        pending_ids = []
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 1. Kiểm tra cache đã tồn tại
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

        print(f"    [*] Generating {len(pending_problems)} problems locally using Hugging Face batching (Batch Size: {batch_size})...")
        
        from tqdm import tqdm
        
        # 2. Thực hiện sinh theo Lô (Batch Generation)
        for i in tqdm(range(0, len(pending_problems), batch_size), desc="Hugging Face Batches"):
            batch_probs = pending_problems[i : i + batch_size]
            batch_ids = pending_ids[i : i + batch_size]
            
            # Xây dựng prompt cho từng phần tử trong lô
            prompts = []
            for prob in batch_probs:
                user_content = f"{GENERATION_PROMPT}\n\nProblem:\n{prob}"
                messages = [{"role": "user", "content": user_content}]
                prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_text += "Step 1:"
                prompts.append(prompt_text)
                
            # Tokenize và đưa lên CUDA
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            input_len = inputs.input_ids.shape[1]
            
            try:
                # Sinh đồng loạt K câu trả lời cho cả Lô
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        num_return_sequences=num_rollouts,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Giải mã kết quả (Shape của outputs: [Len(batch_probs) * num_rollouts, SeqLen])
                for idx, p_id in enumerate(batch_ids):
                    problem_rollouts = []
                    for r in range(num_rollouts):
                        out_idx = idx * num_rollouts + r
                        out_tokens = outputs[out_idx]
                        gen_text = "Step 1: " + self.tokenizer.decode(out_tokens[input_len:], skip_special_tokens=True).strip()
                        problem_rollouts.append(gen_text)
                        
                    results[p_id] = problem_rollouts
                    
                    # Ghi Cache từng bài toán để đảm bảo an toàn dữ liệu
                    if cache_dir:
                        cache_file = os.path.join(cache_dir, f"{p_id}.json")
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(problem_rollouts, f, ensure_ascii=False, indent=2)
                            
            except Exception as e:
                print(f"    [!] Error generating batch starting at index {i}: {e}")
                for p_id in batch_ids:
                    results[p_id] = [""] * num_rollouts
                    
        print(f"    [+] Batch generation completed.")
        return results

    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens, batch_size=1)
        return res.get(problem_id, [])
