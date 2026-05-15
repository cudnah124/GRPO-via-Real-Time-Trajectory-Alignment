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
        self.rollout_batch_size = 4
        
        # Sử dụng bfloat16 và ép lên GPU cuda:0 để tối ưu tốc độ
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa"
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Bắt buộc padding_side="left" cho batch generation
        self.tokenizer.padding_side = "left"

        print(f"[*] Model loaded on: {next(self.model.parameters()).device}")

    @torch.inference_mode()
    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        rollouts = []
        cache_file = None
        
        # 1. Load cache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{problem_id}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        rollouts = json.load(f)
                    print(f"    [+] Loaded {len(rollouts)} rollouts from cache.")
                except Exception:
                    rollouts = []

        if len(rollouts) >= num_rollouts:
            return rollouts[:num_rollouts]

        needed = num_rollouts - len(rollouts)
        messages = [
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": problem}
        ]

        # 2. Chuẩn bị input (trả về BatchEncoding)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = input_ids.shape[-1]

        # 3. Sinh theo batch
        remaining = needed
        print(f"    [*] Generating {needed} rollouts (batch_size={self.rollout_batch_size})...")
        
        while remaining > 0:
            batch_size = min(remaining, self.rollout_batch_size)
            
            # Repeat input cho batch
            batch_input_ids = input_ids.repeat(batch_size, 1)
            batch_attention_mask = attention_mask.repeat(batch_size, 1)

            try:
                outputs = self.model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Giải mã
                for output in outputs:
                    gen_text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                    rollouts.append(gen_text)
                    print(f"    [+] Decoded rollout {len(rollouts)}/{num_rollouts}")
                
                remaining -= batch_size
                
                # Lưu cache ngay lập tức
                if cache_file:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(rollouts, f, ensure_ascii=False, indent=2)
                        
            except torch.cuda.OutOfMemoryError:
                # ✅ Thêm block này
                torch.cuda.empty_cache()
                self.rollout_batch_size = max(1, batch_size // 2)
                print(f"    [!] OOM — giảm batch_size xuống {self.rollout_batch_size}")
                continue  # Thử lại với batch nhỏ hơn

            except Exception as e:
                print(f"    [!] Generation Error: {e}")
                break
            
        return rollouts