import os
import json
import re
import random
import asyncio
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm_asyncio

# Cấu hình GCP và Client giống judge_demo.py
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "model-ruler-497006-p5")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "gemini-2.5-flash")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# Các prompts cho từng loại đột biến dựa trên data_generation_guide.md
PROMPTS = {
    0.05: {
        "system": "You are a mathematical writing assistant. Your task is to rewrite the given correct multi-step solution. Change the connecting words (e.g., 'therefore', 'hence'), simplify or expand the verbal explanations, but keep the underlying mathematical equations and the logical steps 100% identical and correct.",
        "template": "Input: {rollout}\nOutput: "
    },
    0.40: {
        "system": "You are a mathematical code augmentor. Your task is to take a correct multi-step math solution (LaTeX formatting) and introduce EXACTLY ONE realistic mathematical or arithmetic mistake (e.g., sign inversion, incorrect addition, wrong exponent simplification) in the intermediate steps. Ensure the rest of the writing flow is natural and grammatically correct. Do NOT make the final answer correct.",
        "template": "Input: {rollout}\nOutput: "
    },
    0.65: {
        "system": "You are a mathematical code augmentor. Your task is to take a correct multi-step math solution and truncate it. Either stop the explanation abruptly before concluding the final answer, or completely skip 2 intermediate steps and directly output an unproven or wrong final answer.",
        "template": "Input: {rollout}\nOutput: "
    }
}

# Semaphore giới hạn tối đa 15 yêu cầu đồng thời lên API để tránh 429
sem = asyncio.Semaphore(15)

async def call_gemini_mutator(rollout, target_severity):
    config_data = PROMPTS[target_severity]
    system_instruction = config_data["system"]
    prompt = config_data["template"].format(rollout=rollout)
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7, # Cho phép tự do sáng tạo nhẹ để tạo lỗi tự nhiên
        max_output_tokens=2048
    )
    
    max_retries = 5
    base_delay = 2.0
    
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=config
                )
                txt = response.text.strip()
                if txt:
                    return txt
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    break
        return None

# Cấu hình Flag chỉ chạy ghép chéo bài toán (Cross-Problem), KHÔNG gọi API Gemini
CROSS_ONLY = True

async def process_item(item, cycle_idx, all_other_rollouts, cross_only=CROSS_ONLY):
    rollouts = item.get('generated_rollouts', [])
    costs = item.get('alignment_costs', {})
    
    if not rollouts or not isinstance(rollouts, list) or len(rollouts) == 0:
        return item

    # 1. Tạo các đột biến ngoại tuyến bằng Gemini API (Chỉ chạy khi cross_only = False)
    if not cross_only:
        severity_cycle = [0.05, 0.40, 0.65]
        num_base_rollouts = len(rollouts)
        
        for offset in range(2):
            idx_to_mutate = offset % num_base_rollouts
            base_rollout = rollouts[idx_to_mutate]
            
            target_severity = severity_cycle[(cycle_idx + offset) % 3]
            
            # Gọi Gemini sinh đột biến text
            mutated_rollout = await call_gemini_mutator(base_rollout, target_severity)
            
            if mutated_rollout:
                mutated_idx = len(rollouts)
                rollouts.append(mutated_rollout)
                
                # Gán nhãn cứng so sánh giữa rollout gốc và rollout đột biến (đều có khoảng cách target_severity ở mọi ô)
                pair_key = f"({idx_to_mutate},{mutated_idx})"
                costs[pair_key] = target_severity
                
                # Gán nhãn cứng giữa rollout đột biến mới và các rollout đúng còn lại
                for other_idx in range(len(rollouts) - 1):
                    if other_idx != idx_to_mutate:
                        m_key = f"({min(other_idx, mutated_idx)},{max(other_idx, mutated_idx)})"
                        costs[m_key] = target_severity

    # 2. Tạo 1 cặp Lạc đề hoàn toàn (Cross-Problem - Severity 1.0) HOÀN TOÀN KHÔNG DÙNG API
    if all_other_rollouts:
        current_set = set(rollouts)
        valid_others = [r for r in all_other_rollouts if r not in current_set]
        
        if valid_others:
            cross_rollout = random.choice(valid_others)
            cross_idx = len(rollouts)
            rollouts.append(cross_rollout)
            
            # Gán nhãn cứng cho tất cả các cặp so sánh với rollout lạc đề này là 1.0
            for other_idx in range(len(rollouts) - 1):
                m_key = f"({min(other_idx, cross_idx)},{max(other_idx, cross_idx)})"
                costs[m_key] = 1.0
                    
    item['generated_rollouts'] = rollouts
    item['alignment_costs'] = costs
    return item

async def main_async():
    input_path = "distillation_data.jsonl"
    output_path = "distillation_data_augmented.jsonl"
    
    if not os.path.exists(input_path):
        print(f"[!] Lỗi: Không tìm thấy file gốc '{input_path}'. Hãy chạy python package_dataset.py trước.")
        return
        
    print(f"[*] Đang đọc file gốc '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
        
    # Gom tất cả các rollout của các bài toán khác để phục vụ ghép chéo bài toán (Cross-Problem)
    all_other_rollouts = []
    for item in items:
        all_other_rollouts.extend(item.get('generated_rollouts', []))
        
    if CROSS_ONLY:
        print("[*] Chế độ: Chỉ chạy ghép chéo bài toán (Cross-Problem) - KHÔNG sử dụng API.")
    else:
        print("[*] Chế độ: Huấn luyện hỗn hợp - Sử dụng Gemini API sinh đột biến + Ghép chéo bài toán.")
        
    print(f"[*] Đang tiến hành tăng cường dữ liệu...")
    tasks = []
    for idx, item in enumerate(items):
        tasks.append(process_item(item, idx * 2, all_other_rollouts))
        
    augmented_items = await tqdm_asyncio.gather(*tasks, desc="Đang sinh đột biến bằng Gemini")
    
    print(f"[*] Đang lưu tập dữ liệu tăng cường vào '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("="*60)
    print("HOÀN TẤT TĂNG CƯỜNG DỮ LIỆU!")
    print(f"[+] File kết quả tăng cường: {os.path.abspath(output_path)}")
    print(f"[+] Đã hoàn thành xử lý cho {len(augmented_items)} bài toán.")
    print("="*60)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
