import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID, problem_batch_size=4):
        print(f"[*] Loading local model: {model_id}...")
        self.model_id = model_id
        self.problem_batch_size = problem_batch_size  # Số bài xử lý song song mặc định
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Tối ưu cho GPU T4 (bf16/fp16 + SDPA)
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

    def _generate_chunk(self, chunk_problems, chunk_ids, cache_dir, num_rollouts, max_tokens):
        """Sinh rollouts cho 1 chunk (problem_batch_size bài)."""
        results = {}

        # Tokenize cả chunk
        batch_prompts = []
        for prob in chunk_problems:
            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": prob}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            batch_prompts.append(prompt_text)

        inputs = self.tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[-1]

        try:
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=num_rollouts,
                pad_token_id=self.tokenizer.pad_token_id
            )

            for i, p_id in enumerate(chunk_ids):
                start = i * num_rollouts
                end   = start + num_rollouts
                rollouts = [
                    self.tokenizer.decode(outputs[j][prompt_len:], skip_special_tokens=True)
                    for j in range(start, end)
                ]
                results[p_id] = rollouts

                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"{p_id}.json")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(rollouts, f, ensure_ascii=False, indent=2)

            print(f"    [+] Chunk done: {len(chunk_ids)} problems.")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Chia đôi chunk và thử lại (Recursive)
            if len(chunk_problems) == 1:
                print(f"    [!] OOM trên 1 bài duy nhất — bỏ qua {chunk_ids[0]}")
                return {chunk_ids[0]: []}
            mid = len(chunk_problems) // 2
            print(f"    [!] OOM — chia chunk thành {mid} + {len(chunk_problems)-mid}")
            results.update(self._generate_chunk(chunk_problems[:mid], chunk_ids[:mid], cache_dir, num_rollouts, max_tokens))
            results.update(self._generate_chunk(chunk_problems[mid:], chunk_ids[mid:], cache_dir, num_rollouts, max_tokens))

        except Exception as e:
            print(f"    [!] Chunk Error: {e}")

        return results

    @torch.inference_mode()
    def generate_batch(self, problems, problem_ids, cache_dir=None,
                       num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        """Xử lý danh sách bài toán lớn bằng cách chia nhỏ thành các chunks."""
        results = {}
        pending_problems = []
        pending_ids = []

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 1. Lọc những bài đã có trong cache
        for prob, p_id in zip(problems, problem_ids):
            cache_file = os.path.join(cache_dir, f"{p_id}.json") if cache_dir else None
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if len(cached) >= num_rollouts:
                        results[p_id] = cached[:num_rollouts]
                        continue
                except Exception:
                    pass
            pending_problems.append(prob)
            pending_ids.append(p_id)

        if not pending_problems:
            return results

        # 2. Chia thành các chunks nhỏ hơn để tránh OOM tổng thể
        total = len(pending_problems)
        for chunk_start in range(0, total, self.problem_batch_size):
            chunk_end = min(chunk_start + self.problem_batch_size, total)
            chunk_problems = pending_problems[chunk_start:chunk_end]
            chunk_ids      = pending_ids[chunk_start:chunk_end]

            print(f"    [*] Chunk {chunk_start//self.problem_batch_size + 1}: "
                  f"{len(chunk_problems)} bài × {num_rollouts} rollouts")

            chunk_results = self._generate_chunk(
                chunk_problems, chunk_ids, cache_dir, num_rollouts, max_tokens
            )
            results.update(chunk_results)

        return results

    def generate(self, problem, problem_id, cache_dir=None,
                 num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens)
        return res.get(problem_id, [])