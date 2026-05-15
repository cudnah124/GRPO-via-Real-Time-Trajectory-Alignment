import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

# Tune batch size theo VRAM của máy
ROLLOUT_BATCH_SIZE = 4   # 8GB VRAM → 2-4 | 24GB → 8+

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Loading local model: {model_id}...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # ✅ Fix 1: Ép dtype về bfloat16, không dùng "auto"
        # ✅ Fix 2: Chỉ load lên GPU (cuda:0), tránh CPU offload
        # ✅ Fix 3: flash_attention_2 nhanh hơn sdpa ~20-40%
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,          # hoặc float16 nếu GPU cũ không hỗ trợ bf16
            device_map="cuda:0",                  # Explicit GPU, không để "auto" tự split
            attn_implementation="flash_attention_2"  # pip install flash-attn --no-build-isolation
        )
        self.model.eval()  # ✅ Fix 4: Tắt dropout layers

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ✅ Fix 5: padding_side="left" bắt buộc khi batch generate
        self.tokenizer.padding_side = "left"

        print(f"[*] Model loaded on: {next(self.model.parameters()).device}")
        print(f"[*] Model dtype: {next(self.model.parameters()).dtype}")

    @torch.inference_mode()
    def generate(self, problem, problem_id, cache_dir=None,
                 num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        rollouts = []
        cache_file = None

        # Load cache
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

        # Chuẩn bị 1 lần, dùng lại cho tất cả batches
        single_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        prompt_len = single_input.shape[-1]

        # ✅ Fix 6: Sinh theo batch nhỏ thay vì tất cả cùng lúc
        # → tránh OOM, throughput thực tế cao hơn
        print(f"    [*] Generating {needed} rollouts (batch_size={ROLLOUT_BATCH_SIZE})...")
        
        remaining = needed
        while remaining > 0:
            batch_size = min(remaining, ROLLOUT_BATCH_SIZE)

            # ✅ Fix 7: Repeat input cho đúng batch_size, padding left
            batch_input_ids = single_input.repeat(batch_size, 1).to(self.model.device)
            attention_mask = torch.ones_like(batch_input_ids)

            try:
                outputs = self.model.generate(
                    batch_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1,   # ✅ Fix 8: =1 vì đã repeat input thủ công
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,           # ✅ KV-cache (default=True, để tường minh)
                )

                for output in outputs:
                    gen_text = self.tokenizer.decode(
                        output[prompt_len:], skip_special_tokens=True
                    )
                    rollouts.append(gen_text)
                    print(f"    [+] Rollout {len(rollouts)}/{num_rollouts} done")

                remaining -= batch_size

            except torch.cuda.OutOfMemoryError:
                # ✅ Fix 9: Tự động giảm batch khi OOM thay vì crash
                print(f"    [!] OOM at batch_size={batch_size}, reducing to {batch_size // 2}...")
                global ROLLOUT_BATCH_SIZE
                ROLLOUT_BATCH_SIZE = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue

            except Exception as e:
                print(f"    [!] Generation Error: {e}")
                break

        # Lưu cache
        if cache_file and rollouts:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(rollouts, f, ensure_ascii=False, indent=2)

        if len(rollouts) < num_rollouts:
            print(f"    [!] Not enough rollouts ({len(rollouts)}/{num_rollouts}).")
            return []

        return rollouts[:num_rollouts]