import requests
import time
from phase1_distillation.prompts import GENERATION_PROMPT
import phase1_distillation.config as config

class MathRolloutGenerator:
    def __init__(self, model_id=config.MODEL_ID):
        self.model_id = model_id
        print(f"Initialized API Generator with model: {self.model_id}")

    def generate(self, problem, num_rollouts=config.K_ROLLOUTS, max_tokens=1024):
        rollouts = []
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": problem}
            ],
            "temperature": 0.7, 
            "max_tokens": max_tokens
        }
        
        # Gọi lặp để đảm bảo lấy đủ K rollouts (do API có thể không hỗ trợ tham số `n` ổn định)
        for i in range(num_rollouts):
            try:
                response = requests.post(config.API_URL, headers=config.HEADERS, json=payload, timeout=90)
                if response.status_code == 200:
                    gen_text = response.json()["choices"][0]["message"]["content"]
                    rollouts.append(gen_text)
                else:
                    print(f"    [!] API Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"    [!] Request Failed: {e}")
                
            time.sleep(1) # Tránh Rate Limit
            
        return rollouts
