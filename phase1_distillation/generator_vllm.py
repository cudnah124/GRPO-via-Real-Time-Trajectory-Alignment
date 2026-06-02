import os
import json
import torch
from transformers import AutoTokenizer
import phase1_distillation.config as config
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id=config.GENERATOR_MODEL_ID):
        print(f"[*] Initializing local vLLM Generator with: {model_id}...")
        self.model_id = model_id
        
        # Thêm đường dẫn CUDA 12.8 vào LD_LIBRARY_PATH để hệ thống tìm thấy libcudart.so.13 vừa liên kết
        cuda_lib_dir = "/usr/local/cuda-12.8/lib64"
        if "LD_LIBRARY_PATH" in os.environ:
            if cuda_lib_dir not in os.environ["LD_LIBRARY_PATH"]:
                os.environ["LD_LIBRARY_PATH"] += f":{cuda_lib_dir}"
        else:
            os.environ["LD_LIBRARY_PATH"] = cuda_lib_dir
            
        # Tải tokenizer cho mục đích tương thích ngược
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Lazy import vLLM
        try:
            from vllm import LLM, SamplingParams
            self.vllm_class = LLM
            self.sampling_params_class = SamplingParams
        except ImportError as e:
            print(f"\n[!] LỖI HỆ THỐNG GỐC KHI IMPORT vLLM: {e}")
            import traceback
            traceback.print_exc()
            raise ImportError(
                f"Không thể import vLLM do lỗi hệ thống: {e}\n"
                "Vui lòng kiểm tra lại cấu hình CUDA hoặc liên kết driver."
            )
        
        # Vá lỗi fileno cho môi trường Colab
        import sys
        if not hasattr(sys.stdout, 'fileno'):
            sys.stdout.fileno = lambda: 1
        if not hasattr(sys.stderr, 'fileno'):
            sys.stderr.fileno = lambda: 2
            
        # Cấu hình vLLM tối ưu cho GPU của Colab
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=0.85,  # Dành 85% VRAM cho Generator
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,          # Bắt buộc chế độ eager để tránh treo/OOM CUDA
            disable_log_stats=True,
            hf_overrides={"rope_scaling": None}  # Bỏ qua lỗi rope_scaling trong transformers mới
        )

    def generate_batch(self, problems, problem_ids, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
        results = {}
        pending_problems = []
        pending_ids = []
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 1. Kiểm tra cache đã tồn tại
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

        # 2. Định nghĩa tham số sinh (n = num_rollouts sinh song song K kết quả)
        sampling_params = self.sampling_params_class(
            n=num_rollouts,
            temperature=0.7,
            max_tokens=max_tokens,
            skip_special_tokens=True
        )

        # Cấu hình kích thước sub-batch để lưu cache liên tục
        sub_batch_size = 32
        print(f"    [*] vLLM processing {len(pending_problems)} problems in sub-batches of size {sub_batch_size}...")
        
        from tqdm import tqdm
        tokenizer = self.llm.get_tokenizer()
        
        # 3. Chạy chia lô nhỏ để ghi cache liên tục phòng trường hợp ngắt kết nối Colab
        for i in tqdm(range(0, len(pending_problems), sub_batch_size), desc="vLLM Generating Rollouts"):
            batch_probs = pending_problems[i : i + sub_batch_size]
            batch_ids = pending_ids[i : i + sub_batch_size]
            
            prompts = []
            for prob in batch_probs:
                user_content = f"{GENERATION_PROMPT}\n\nProblem:\n{prob}"
                messages = [{"role": "user", "content": user_content}]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_text += "Step 1:"
                prompts.append(prompt_text)
                
            try:
                outputs = self.llm.generate(prompts, sampling_params)
                
                for p_id, output in zip(batch_ids, outputs):
                    problem_rollouts = ["Step 1: " + out.text.strip() for out in output.outputs]
                    results[p_id] = problem_rollouts
                    
                    # Lưu Cache cho bài toán này ngay lập tức
                    if cache_dir:
                        cache_file = os.path.join(cache_dir, f"{p_id}.json")
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(problem_rollouts, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"    [!] Error generating sub-batch: {e}")
                for p_id in batch_ids:
                    results[p_id] = [""] * num_rollouts
                    
        print(f"    [+] vLLM generation batch completed successfully.")
        return results

    def generate(self, problem, problem_id, cache_dir=None, num_rollouts=config.K_ROLLOUTS, max_tokens=2048):
        res = self.generate_batch([problem], [problem_id], cache_dir, num_rollouts, max_tokens)
        return res.get(problem_id, [])
