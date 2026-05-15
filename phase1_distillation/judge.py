import os
import time
import json
import os
import itertools
from openai import OpenAI
from phase1_distillation.prompts import JUDGE_PROMPT, RETRY_PROMPT
from phase1_distillation.parser import parse_distance_matrix
import phase1_distillation.config as config

class AlignmentJudge:
    def __init__(self, model_id=config.JUDGE_MODEL_ID):
        self.model_id = model_id

    def _get_client(self):
        from phase1_distillation.client_manager import rotator
        return rotator.get_client()

    def evaluate_single_pair(self, problem, rollout_a, rollout_b):
        """Evaluate a single pair using OpenAI client with retry logic"""
        content = f"Problem: {problem}\n\nRollout A:\n{rollout_a}\n\nRollout B:\n{rollout_b}"
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": content}
        ]
        
        for attempt in range(config.MAX_RETRIES):
            client = self._get_client() # Di chuyển vào trong loop để đổi Token nếu lỗi
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

    def evaluate_pairs(self, problem, problem_id, rollouts, cache_dir=None):
        """Evaluate all combinations C(K,2) with incremental caching"""
        results = {}
        if len(rollouts) < 2:
            return results
            
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"judge_{problem_id}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    print(f"    [+] Loaded {len(results)} pairs from judge cache.")
                except:
                    results = {}

        indices = list(range(len(rollouts)))
        pairs = list(itertools.combinations(indices, 2))
        
        for i, j in pairs:
            pair_key = f"({i},{j})"
            if pair_key in results and results[pair_key] is not None:
                continue
                
            print(f"    [*] Evaluating pair {pair_key}...")
            matrix = self.evaluate_single_pair(problem, rollouts[i], rollouts[j])
            
            if matrix is None:
                print(f"    [!] Critical: Pair {pair_key} failed. Saving partial progress.")
                if cache_file:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                return None
                
            results[pair_key] = matrix
            
            if cache_file:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results
