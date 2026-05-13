import os
import time
from openai import OpenAI
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.MODEL_ID):
        self.model_id = model_id

    def _get_client(self):
        from phase1_distillation.client_manager import rotator
        return rotator.get_client()

    def generate(self, problem, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        rollouts = []
        max_attempts = num_rollouts * 2 # Thử tối đa gấp đôi số lượng cần thiết
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
            except Exception as e:
                print(f"    [!] Generation Attempt {attempts} failed: {e}")
                time.sleep(2)
                
            time.sleep(1)
            
        # Nếu không đủ K câu trả lời, trả về list rỗng để main loop skip (và retry sau này)
        if len(rollouts) < num_rollouts:
            print(f"    [!] Failed to generate enough rollouts ({len(rollouts)}/{num_rollouts})")
            return []
            
        return rollouts
