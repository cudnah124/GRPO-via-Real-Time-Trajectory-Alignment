import os
import time
import json
import os
from openai import OpenAI
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.MODEL_ID):
        self.model_id = model_id

    def _get_client(self):
        from phase1_distillation.client_manager import rotator
        return rotator.get_client()

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
        max_attempts = num_rollouts * 2
        attempts = 0
        
        while len(rollouts) < num_rollouts and attempts < max_attempts:
            attempts += 1
            client = self._get_client()
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": GENERATION_PROMPT},
                        {"role": "user", "content": problem}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                gen_text = response.choices[0].message.content
                rollouts.append(gen_text)
                print(f"    [+] Generated rollout {len(rollouts)}/{num_rollouts}")
                
                # Lưu vào cache ngay lập tức sau mỗi lần sinh thành công
                if cache_file:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(rollouts, f, ensure_ascii=False, indent=2)
                        
            except Exception as e:
                print(f"    [!] Generation Attempt {attempts} failed: {e}")
                time.sleep(2)
                
            time.sleep(1)
            
        if len(rollouts) < num_rollouts:
            print(f"    [!] Still not enough rollouts ({len(rollouts)}/{num_rollouts}). Keeping cache for next time.")
            return []
            
        return rollouts
