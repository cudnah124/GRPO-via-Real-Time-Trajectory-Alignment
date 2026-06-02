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

from phase1_distillation.prompts import JUDGE_PROMPT

async def evaluate_alignment_genai_async(rollout_a, rollout_b):
    import re
    def get_steps_list(text):
        parts = re.split(r'(?i)(Step\s+\d+:)', text)
        steps = []
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            steps.append(f"{header} {content}")
        if not steps and text.strip():
            steps = [s.strip() for s in text.split("\n\n") if s.strip()]
        return steps

    steps_a = get_steps_list(rollout_a)
    steps_b = get_steps_list(rollout_b)
    
    N = len(steps_a)
    M = len(steps_b)

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
        system_instruction=JUDGE_PROMPT,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=dist_matrix_schema,
        thinking_config=types.ThinkingConfig(thinking_budget=512),
        max_output_tokens=4096
    )
    
    # Định dạng prompt rõ ràng kèm kích thước ma trận yêu cầu
    prompt = f"""Compare step-by-step logical alignment between:
Rollout A (has {N} steps):
{chr(10).join(f'- A{i+1}: {step}' for i, step in enumerate(steps_a))}

Rollout B (has {M} steps):
{chr(10).join(f'- B{j+1}: {step}' for j, step in enumerate(steps_b))}

You MUST return a 2D JSON array of size exactly {N} x {M} (containing {N} rows, where each row has exactly {M} numbers between 0.0 and 1.0).
Do NOT explain. Do NOT add any text other than the JSON array."""

    # Cơ chế Retry tự động với Exponential Backoff khi dính lỗi 429
    max_retries = 5
    base_delay = 2.0  # Giờ chờ ban đầu: 2 giây
    
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            # In lượng token sử dụng (bao gồm cả thinking/candidates token)
            if response.usage_metadata:
                meta = response.usage_metadata
                print(f"    [Tokens] Input: {meta.prompt_token_count} | Output (gồm cả Thinking nếu có): {meta.candidates_token_count} | Total: {meta.total_token_count}")
            try:
                return json.loads(response.text)
            except json.JSONDecodeError as jde:
                print(f"[!] Lỗi giải mã JSON: {jde}")
                print(f"    [Raw Response]: {repr(response.text)}")
                return None
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                delay = base_delay * (2 ** attempt)
                print(f"    [!] Gặp lỗi Rate Limit (429). Thử lại lần {attempt + 1}/{max_retries} sau {delay:.1f} giây...")
                await asyncio.sleep(delay)
            else:
                print(f"[!] Lỗi gọi API: {e}")
                return None
    print("    [!] Thất bại sau 5 lần thử lại do liên tục dính lỗi 429.")
    return None

async def main_async():
    # 1. Cấu hình thư mục đầu vào và đầu ra cục bộ
    input_dir = os.path.join("data", "rollouts")
    output_dir = os.path.join("data", "judge")
    
    if not os.path.exists(input_dir):
        print(f"[!] Thư mục đầu vào không tồn tại: {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)

    # 2. Quét thư mục rollout cục bộ tìm các file json
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    print(f"[+] Tìm thấy {len(json_files)} file rollouts trong thư mục {input_dir}.")
    
    # Lọc bỏ các file đã hoàn thành đầy đủ
    pending_files = []
    for filename in json_files:
        output_file_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_file_path):
            pending_files.append(filename)
            continue
            
        # Nếu file đã tồn tại, kiểm tra xem có ma trận nào bị trống (hoặc thiếu) không
        try:
            input_file_path = os.path.join(input_dir, filename)
            with open(input_file_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)
            
            # Lấy số lượng rollouts
            rollouts = []
            if isinstance(input_data, dict):
                for key in ["rollouts", "generated_rollouts", "responses", "answers", "outputs"]:
                    if key in input_data and isinstance(input_data[key], list):
                        rollouts = input_data[key]
                        break
            elif isinstance(input_data, list):
                rollouts = input_data
                
            num_rollouts = len(rollouts)
            if num_rollouts < 2:
                continue
                
            # Đọc file kết quả hiện tại
            with open(output_file_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            
            existing_matrices = output_data.get("distance_matrices", {})
            
            # Tính toán các cặp dự kiến
            expected_pairs = 0
            valid_pairs = 0
            for r_a, r_b in itertools.combinations(range(num_rollouts), 2):
                expected_pairs += 1
                key = f"({r_a},{r_b})"
                matrix = existing_matrices.get(key)
                if matrix and len(matrix) > 0:
                    valid_pairs += 1
            
            if valid_pairs < expected_pairs:
                pending_files.append(filename)
        except Exception:
            # Nếu có lỗi khi đọc/parse, coi như cần xử lý lại từ đầu
            pending_files.append(filename)
            
    print(f"[*] Có {len(pending_files)} / {len(json_files)} bài toán chưa hoàn thành hoàn toàn.")
    if not pending_files:
        print("[+] Tất cả bài toán đã được xử lý xong.")
        return
 
    # Cấu hình kích thước lô (Batch Size) - Cấu hình 5 câu/lượt theo yêu cầu
    BATCH_SIZE = 3
    print(f"[*] Bắt đầu xử lý song song với cấu hình: {BATCH_SIZE} câu/lượt...")
    
    # 3. Xử lý chia lô (Batching)
    for idx in range(0, len(pending_files), BATCH_SIZE):
        batch_files = pending_files[idx : idx + BATCH_SIZE]
        print(f"\n" + "="*60)
        print(f"BẮT ĐẦU BATCH {idx // BATCH_SIZE + 1} (Gồm {len(batch_files)} bài toán)")
        print("="*60)
        
        batch_tasks = []
        batch_meta = [] # Lưu thông tin meta để phân bổ kết quả sau khi gather
        
        # Đọc dữ liệu và tạo tasks song song cho cả Batch
        for filename in batch_files:
            p_id = os.path.splitext(filename)[0]
            file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[!] Lỗi đọc file {filename}: {e}")
                continue
                
            # Xác định các rollouts trực tiếp từ nội dung file JSON
            rollouts = []
            if isinstance(data, dict):
                # Quét các key chứa rollouts
                for key in ["rollouts", "generated_rollouts", "responses", "answers", "outputs"]:
                    if key in data and isinstance(data[key], list):
                        rollouts = data[key]
                        break
            elif isinstance(data, list):
                # Trường hợp file chỉ lưu mảng (list), mặc định các phần tử trong list là các rollout
                rollouts = data
                
            if len(rollouts) < 2:
                print(f"[!] ID {p_id} có ít hơn 2 rollouts. Bỏ qua...")
                continue
                
            # Đọc các ma trận đã tồn tại nếu có
            existing_matrices = {}
            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        old_record = json.load(f)
                        existing_matrices = old_record.get("distance_matrices", {})
                except Exception:
                    pass

            # Tạo các cặp đối sánh
            indices = list(range(len(rollouts)))
            pair_keys = []
            pair_tasks = []
            
            for r_a, r_b in itertools.combinations(indices, 2):
                key = f"({r_a},{r_b})"
                existing_matrix = existing_matrices.get(key)
                if existing_matrix and len(existing_matrix) > 0:
                    # Đã có kết quả hợp lệ, giữ nguyên
                    pass
                else:
                    # Chưa có kết quả hoặc rỗng, cần gọi API
                    pair_keys.append(key)
                    pair_tasks.append(evaluate_alignment_genai_async(rollouts[r_a], rollouts[r_b]))
                
            # Gom thông tin lại để giải nén kết quả sau đó
            batch_meta.append({
                "p_id": p_id,
                "filename": filename,
                "rollouts": rollouts,
                "existing_matrices": existing_matrices,
                "pair_keys": pair_keys,
                "task_count": len(pair_tasks)
            })
            batch_tasks.extend(pair_tasks)
            
        if not batch_tasks:
            # Có thể trong batch này toàn các file đã hoàn thành (nhưng chưa lưu trạng thái đúng?)
            # Hoặc tất cả các cặp đã được tính toán. Ta vẫn cập nhật để chắc chắn.
            for meta in batch_meta:
                output_file_path = os.path.join(output_dir, meta["filename"])
                # Lưu lại cho chắc
                record = {
                    "problem_id": meta["p_id"],
                    "generated_rollouts": meta["rollouts"],
                    "distance_matrices": meta["existing_matrices"]
                }
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(compact_json_format(record) + "\n")
            continue
            
        print(f"[*] Đang gửi song song {len(batch_tasks)} yêu cầu đánh giá lên Gemini...")
        all_results = await asyncio.gather(*batch_tasks)
        
        # Phân bổ lại kết quả cho từng bài toán trong Batch và ghi file
        result_offset = 0
        for meta in batch_meta:
            p_id = meta["p_id"]
            filename = meta["filename"]
            rollouts = meta["rollouts"]
            existing_matrices = meta["existing_matrices"]
            pair_keys = meta["pair_keys"]
            task_count = meta["task_count"]
            
            prob_results = all_results[result_offset : result_offset + task_count]
            result_offset += task_count
            
            # Khởi tạo từ kết quả cũ
            distance_matrices = dict(existing_matrices)
            for key, matrix in zip(pair_keys, prob_results):
                if matrix:
                    distance_matrices[key] = matrix
                elif key not in distance_matrices:
                    distance_matrices[key] = []
                
            # Tạo bản ghi kết quả
            record = {
                "problem_id": p_id,
                "generated_rollouts": rollouts,
                "distance_matrices": distance_matrices
            }
            
            # Hàm format Custom để giữ ma trận gọn gàng trên 1 dòng
            def compact_json_format(data_dict):
                formatted_matrices = {}
                for key, matrix in data_dict["distance_matrices"].items():
                    if matrix:
                        matrix_lines = []
                        for row in matrix:
                            matrix_lines.append("[" + ", ".join(f"{val:.1f}" for val in row) + "]")
                        formatted_matrices[key] = "[\n      " + ",\n      ".join(matrix_lines) + "\n    ]"
                    else:
                        formatted_matrices[key] = "[]"
                
                temp_record = {k: v for k, v in data_dict.items() if k != "distance_matrices"}
                base_json = json.dumps(temp_record, ensure_ascii=False, indent=2)
                
                matrix_json_parts = []
                for key, formatted_str in formatted_matrices.items():
                    matrix_json_parts.append(f'    "{key}": {formatted_str}')
                
                matrix_block = "{\n" + ",\n".join(matrix_json_parts) + "\n  }"
                json_str = base_json.rstrip().rstrip("}") + f',\n  "distance_matrices": {matrix_block}\n}}'
                return json_str

            output_file_path = os.path.join(output_dir, filename)
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(compact_json_format(record) + "\n")
            
        # Nghỉ dài 30 giây giữa các lượt để reset hạn mức Quota của API
        print("[*] Đang nghỉ 30 giây bảo vệ Rate Limit...")
        await asyncio.sleep(70)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
