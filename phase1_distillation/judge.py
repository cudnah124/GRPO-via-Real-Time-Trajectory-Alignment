import os
import time
import itertools
from openai import OpenAI
from phase1_distillation.prompts import JUDGE_PROMPT, RETRY_PROMPT
from phase1_distillation.parser import parse_distance_matrix
import phase1_distillation.config as config

class AlignmentJudge:
    def __init__(self, model_id=config.MODEL_ID):
        self.model_id = model_id

    def _get_client(self):
        from phase1_distillation.client_manager import rotator
        return rotator.get_client()

    def evaluate_single_pair(self, problem, rollout_a, rollout_b):
        """Evaluate a single pair using OpenAI client with retry logic"""
        client = self._get_client()
        content = f"Problem: {problem}\n\nRollout A:\n{rollout_a}\n\nRollout B:\n{rollout_b}"
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": content}
        ]
        
        for attempt in range(config.MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024
                )
                
                raw_output = response.choices[0].message.content
                matrix = parse_distance_matrix(raw_output)
                return matrix
                
            except ValueError as e:
                if attempt < config.MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": RETRY_PROMPT + f"\nError details: {e}"})
                else:
                    print(f"    [!] Failed after {config.MAX_RETRIES} attempts. Error: {e}")
            except Exception as e:
                print(f"    [!] OpenAI Judge Exception: {e}")
                time.sleep(2)
                
            time.sleep(2)
                    
        return None

    def evaluate_pairs(self, problem, rollouts):
        """Evaluate all combinations C(K,2) of rollouts"""
        results = {}
        if len(rollouts) < 2:
            return results
            
        indices = list(range(len(rollouts)))
        pairs = list(itertools.combinations(indices, 2))
        
        for i, j in pairs:
            print(f"    [*] Evaluating pair ({i},{j})...")
            matrix = self.evaluate_single_pair(problem, rollouts[i], rollouts[j])
            results[f"({i},{j})"] = matrix
            
        return results
