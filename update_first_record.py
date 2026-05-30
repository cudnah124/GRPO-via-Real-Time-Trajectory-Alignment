import os
import json
import itertools
import asyncio
from google import genai
from google.genai import types

# Cấu hình dự án GCP và Location từ biến môi trường
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "model-ruler-497006-p5")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "gemini-2.5-flash")

print(f"[*] Khởi tạo Google GenAI Client...")
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# System Prompt cho Judge (Thang điểm liên tục 0.0 - 1.0)
JUDGE_SYSTEM_PROMPT = """You are an expert Logical Alignment Judge for mathematical reasoning trajectories.
Your task is to evaluate the logical alignment step-by-step between two different trajectories (Rollout A and Rollout B) solving the same problem.
You must ignore any differences in vocabulary, grammar, or verbosity. Focus strictly on the LOGICAL and ALGEBRAIC equivalence of the steps.

INSTRUCTIONS FOR GRANULAR SCORING (Scale from 0.0 to 1.0 with step 0.1):
1. Decompose both Rollout A and Rollout B into major logical steps.
2. Compare each step of Rollout A against each step of Rollout B.
3. Evaluate logical similarity on a granular spectrum from 0.0 (identical logic) to 1.0 (completely unrelated/contradictory):
   - 0.0: Perfect mathematical equivalence (even if phrased differently or using different symbols).
   - 0.1 - 0.2: Mathematically equivalent, but uses slightly different notations, or includes a redundant minor step.
   - 0.3 - 0.4: Correct final steps but one uses a different mathematical method, leading to moderate deviation in intermediate steps.
   - 0.5 - 0.6: Minor arithmetic typo (e.g., 2+3=6), notation slip, or a skipped step that does not break the entire proof.
   - 0.7 - 0.8: Severe logical errors, critical step skip, or wrong final answer despite having similar starting steps.
   - 0.9 - 1.0: Completely wrong, contradictory logic, or unrelated mathematical statements.

Output your evaluation purely as a 2D JSON array representing the pairwise distance matrix of size N x M (where N is the number of steps in Rollout A, and M is the number of steps in Rollout B). 
NO EXTRA TEXT. ONLY THE JSON ARRAY."""

async def evaluate_alignment_genai_async(problem, rollout_a, rollout_b):
    dist_matrix_schema = types.Schema(
        type=types.Type.ARRAY,
        description="Ma trận khoảng cách logic 2 chiều kích thước N x M giữa các bước của hai Rollout.",
        items=types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.NUMBER,
                description="Khoảng cách logic từ 0.0 đến 1.0 (bước nhảy 0.1)."
            )
        )
    )
    
    config = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=dist_matrix_schema,
    )
    
    prompt = f"""Problem: {problem}

Rollout A:
{rollout_a}

Rollout B:
{rollout_b}"""

    try:
        # Sử dụng API client không đồng bộ (aio) để gửi song song
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[!] Lỗi gọi API hoặc giải mã JSON: {e}")
        return None

async def main_async():
    jsonl_path = "distillation_data.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"[!] Không tìm thấy file {jsonl_path}.")
        return

    # 1. Đọc toàn bộ nội dung file jsonl
    print("[*] Đang đọc tệp distillation_data.jsonl...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("[!] Tệp rỗng.")
        return

    # 2. Xử lý phần tử đầu tiên
    first_record = json.loads(lines[0])
    rollouts = first_record.get("generated_rollouts", [])
    problem_text = "Prove that points M, N, P, and Q lie on the same circle."
    first_record["problem"] = problem_text

    print(f"[+] Đang chạy song song đánh giá chéo {len(rollouts)} rollouts cho dòng dữ liệu đầu tiên...")
    
    indices = list(range(len(rollouts)))
    tasks = []
    keys = []
    
    # 3. Gom toàn bộ các cặp chéo (0,1), (0,2)... thành các task không đồng bộ
    for i, j in itertools.combinations(indices, 2):
        pair_key = f"({i},{j})"
        keys.append(pair_key)
        tasks.append(evaluate_alignment_genai_async(problem_text, rollouts[i], rollouts[j]))
        
    # 4. Thực thi đồng loạt tất cả các requests cùng lúc
    results = await asyncio.gather(*tasks)
    
    # 5. Phân bổ lại kết quả vào dict
    new_matrices = {}
    for key, matrix in zip(keys, results):
        new_matrices[key] = matrix if matrix else []
            
    # 6. Ghi đè vào record
    first_record["distance_matrices"] = new_matrices
    
    # 7. Cập nhật dòng đầu tiên trong mảng dòng
    lines[0] = json.dumps(first_record, ensure_ascii=False) + "\n"
    
    # 8. Ghi đè lại vào file
    print("[*] Ghi đè dòng đầu tiên vào distillation_data.jsonl...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print("\n[+] ĐÃ THAY ĐỔI DÒNG DỮ LIỆU ĐẦU TIÊN THÀNH CÔNG (PHƯƠNG PHÁP ASYNC PARALLEL)!")
    print("\n=== DÒNG DỮ LIỆU SAU KHI SỬA (PREVIEW JSON) ===")
    preview_data = {
        "problem_id": first_record["problem_id"],
        "problem": first_record["problem"],
        "generated_rollouts_count": len(first_record["generated_rollouts"]),
        "distance_matrices": first_record["distance_matrices"]
    }
    print(json.dumps(preview_data, indent=2, ensure_ascii=False))

def main():
    # Chạy vòng lặp sự kiện asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
