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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa" # Kích hoạt bộ tăng tốc Attention của PyTorch
        )
        # Đảm bảo tokenizer có pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        rollouts = []
        cache_file = None
        
        # 1. Thử load từ cache nếu có
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{problem_id}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        rollouts = json.load(f)
                    print(f"    [+] Loaded {len(rollouts)} rollouts from cache.")
                except:
                    rollouts = []

        # 2. Sinh thêm nếu thiếu
        if len(rollouts) < num_rollouts:
            needed = num_rollouts - len(rollouts)
            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": problem}
            ]
            
            # Chuẩn bị input cho model
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            print(f"    [*] Generating {needed} new rollouts locally...")
            try:
                # Sử dụng num_return_sequences để sinh đồng thời cho nhanh
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=needed,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Giải mã kết quả (chỉ lấy phần text sinh mới)
                prompt_len = input_ids.shape[-1]
                for i, output in enumerate(outputs):
                    gen_text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                    rollouts.append(gen_text)
                    print(f"    [+] Decoded rollout {len(rollouts)}/{num_rollouts}")
                
                # Lưu vào cache sau khi sinh xong
                if cache_file:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(rollouts, f, ensure_ascii=False, indent=2)
                        
                print(f"    [+] Generated {needed} rollouts locally.")
            except Exception as e:
                print(f"    [!] Local Generation Error: {e}")
            
        if len(rollouts) < num_rollouts:
            print(f"    [!] Still not enough rollouts ({len(rollouts)}/{num_rollouts}). Keeping cache for next time.")
            return []
            
        return rollouts
