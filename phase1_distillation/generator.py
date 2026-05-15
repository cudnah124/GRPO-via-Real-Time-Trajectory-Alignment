import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Loading local model: {model_id}...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Tối ưu cho GPU T4
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa"
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # QUAN TRỌNG: padding_side phải là "left" cho batch generation
        self.tokenizer.padding_side = "left"

    @torch.inference_mode()
    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        """
        Xử lý nhiều bài toán cùng lúc để tối ưu GPU throughput.
        problems: list of strings
        problem_ids: list of strings
        """
        results = {}
        pending_problems = []
        pending_ids = []
        
        # 1. Kiểm tra cache và lọc ra những bài cần sinh mới
        for prob, p_id in zip(problems, problem_ids):
            cache_file = os.path.join(cache_dir, f"{p_id}.json") if cache_dir else None
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                        if len(cached_data) >= num_rollouts:
                            results[p_id] = cached_data[:num_rollouts]
                            continue
                except:
                    pass
            
            pending_problems.append(prob)
            pending_ids.append(p_id)
            results[p_id] = [] # Khởi tạo danh sách rỗng

        if not pending_problems:
            return results

        # 2. Chuẩn bị Batch Prompts
        batch_prompts = []
        for prob in pending_problems:
            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": prob}
            ]
            # Dùng apply_chat_template nhưng không tokenize ngay để dễ pad
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_prompts.append(prompt_text)

        # Tokenize cả batch với padding
        inputs = self.tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = input_ids.shape[-1]

        # 3. Sinh rollouts song song
        # Tổng batch thực tế = số bài toán * num_rollouts
        print(f"    [*] Batch Generating: {len(pending_problems)} problems x {num_rollouts} rollouts...")
        
        try:
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=num_rollouts, # Sinh K rollouts cho MỖI câu hỏi
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # 4. Tách kết quả trả về đúng cho từng bài toán
            # Output shape: (len(pending_problems) * num_rollouts, seq_len)
            for i, p_id in enumerate(pending_ids):
                start_idx = i * num_rollouts
                end_idx = start_idx + num_rollouts
                
                problem_rollouts = []
                for j in range(start_idx, end_idx):
                    gen_text = self.tokenizer.decode(outputs[j][prompt_len:], skip_special_tokens=True)
                    problem_rollouts.append(gen_text)
                
                results[p_id] = problem_rollouts
                
                # Lưu vào cache
                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"{p_id}.json")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(problem_rollouts, f, ensure_ascii=False, indent=2)
                        
            print(f"    [+] Batch completed for {len(pending_problems)} problems.")
            
        except Exception as e:
            print(f"    [!] Batch Generation Error: {e}")
            
        return results

    # Giữ lại hàm generate cũ để tương thích (gọi qua generate_batch)
    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens)
        return res.get(problem_id, [])