import os
import time
from openai import OpenAI
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.MODEL_ID):
        self.model_id = model_id

    def _get_client(self):
        return OpenAI(
            base_url=config.API_BASE_URL,
            api_key=os.getenv("HF_TOKEN", config.HF_TOKEN)
        )

    def generate(self, problem, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        client = self._get_client()
        rollouts = []
        
        for i in range(num_rollouts):
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
            except Exception as e:
                print(f"    [!] OpenAI Generation Error: {e}")
                
            time.sleep(1) # Tránh Rate Limit
            
        return rollouts
