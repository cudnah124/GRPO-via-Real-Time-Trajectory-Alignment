import requests
import time
import itertools
from phase1_distillation.prompts import JUDGE_PROMPT, RETRY_PROMPT
from phase1_distillation.parser import parse_distance_matrix
import phase1_distillation.config as config

class AlignmentJudge:
    def __init__(self, model_id=config.MODEL_ID):
        self.model_id = model_id

    def evaluate_single_pair(self, problem, rollout_a, rollout_b):
        """Evaluate a single pair with API retry logic"""
        content = f"Problem: {problem}\n\nRollout A:\n{rollout_a}\n\nRollout B:\n{rollout_b}"
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": content}
        ]
        
        for attempt in range(config.MAX_RETRIES):
            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": 0.2, 
                "max_tokens": 1024
            }
            
            try:
                response = requests.post(config.API_URL, headers=config.HEADERS, json=payload, timeout=90)
                if response.status_code != 200:
                    print(f"    [!] API Error {response.status_code}: {response.text}")
                    time.sleep(2)
                    continue
                    
                raw_output = response.json()["choices"][0]["message"]["content"]
                matrix = parse_distance_matrix(raw_output)
                time.sleep(1) # Tránh Rate limit
                return matrix
                
            except ValueError as e:
                # Nếu parser báo lỗi, gửi lại lỗi cho LLM
                if attempt < config.MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": RETRY_PROMPT + f"\nError details: {e}"})
                else:
                    print(f"    [!] Failed after {config.MAX_RETRIES} attempts. Error: {e}")
            except Exception as e:
                print(f"    [!] Request Exception: {e}")
                time.sleep(2)
                
            time.sleep(1)
                    
        return None

    def evaluate_pairs(self, problem, rollouts):
        """Evaluate all combinations C(K,2) of rollouts"""
        results = {}
        # Nếu không đủ rollouts thì trả về mảng rỗng
        if len(rollouts) < 2:
            return results
            
        indices = list(range(len(rollouts)))
        pairs = list(itertools.combinations(indices, 2))
        
        for i, j in pairs:
            print(f"    [*] Evaluating pair ({i},{j})...")
            matrix = self.evaluate_single_pair(problem, rollouts[i], rollouts[j])
            results[f"({i},{j})"] = matrix
            
        return results
