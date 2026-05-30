import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Initializing native Hugging Face model: {model_id}...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model in Float16 to save memory and run on T4
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
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

        print(f"    [*] Generating {len(pending_problems)} problems locally using transformers...")
        
        from tqdm import tqdm
        
        # Generate step-by-step
        for prob, p_id in tqdm(zip(pending_problems, pending_ids), total=len(pending_problems), desc="Generating rollouts"):
            user_content = f"{GENERATION_PROMPT}\n\nProblem:\n{prob}"
            messages = [
                {"role": "user", "content": user_content}
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_text += "Step 1:"
            
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_len = inputs.input_ids.shape[1]
            
            try:
                # Generate K rollouts
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=num_rollouts,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                )
                
                problem_rollouts = []
                for out in outputs:
                    gen_text = "Step 1: " + self.tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
                    problem_rollouts.append(gen_text)
                
                results[p_id] = problem_rollouts
                
                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"{p_id}.json")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(problem_rollouts, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"    [!] Error generating for {p_id}: {e}")
                results[p_id] = [""] * num_rollouts
                
        print(f"    [+] Batch generation completed.")
        return results

    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens)
        return res.get(problem_id, [])