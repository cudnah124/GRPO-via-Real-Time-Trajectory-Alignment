import json

def create_judge_only_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1.5: Offline Local Judge (High Speed)\n",
                    "Notebook này sử dụng model **Qwen2.5-Math-7B-AWQ** chạy trực tiếp trên T4 để chấm điểm.\n",
                    "\n",
                    "**Lưu ý:** Hãy chắc chắn bạn đã chọn Runtime là **T4 GPU**. Nếu trước đó đã chạy Generator, hãy **Restart Session** để giải phóng VRAM."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Setup & Cài đặt vLLM\n",
                    "!pip install -q vllm jsonlines tqdm transformers\n",
                    "\n",
                    "import os\n",
                    "import json\n",
                    "import jsonlines\n",
                    "import sys\n",
                    "from tqdm import tqdm\n",
                    "from google.colab import drive\n",
                    "\n",
                    "drive.mount('/content/drive')\n",
                    "\n",
                    "# Kiểm tra lại đường dẫn folder Code trên Drive\n",
                    "CODE_PATH = '/content/drive/MyDrive/conference-latex-template/Code'\n",
                    "if CODE_PATH not in sys.path: sys.path.append(CODE_PATH)\n",
                    "\n",
                    "from phase1_distillation import AlignmentJudge\n",
                    "import phase1_distillation.config as config"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Khởi tạo Local Judge Engine\n",
                    "# Model này sẽ chiếm khoảng 5.5GB VRAM. Tốc độ khoảng 1-2 giây/cặp.\n",
                    "judge = AlignmentJudge(model_id=config.JUDGE_MODEL_ID)\n",
                    "\n",
                    "DRIVE_BASE = \"/content/drive/MyDrive/Data_Phase1.5\"\n",
                    "CACHE_DIR = f\"{DRIVE_BASE}/rollouts_cache\"\n",
                    "OUTPUT_FILE = f\"{DRIVE_BASE}/distillation_data.jsonl\"\n",
                    "PROCESSED_IDS_FILE = f\"{DRIVE_BASE}/processed_ids_judge.txt\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Tiến hành chấm điểm hàng loạt (Batch Judging)\n",
                    "processed_ids = set()\n",
                    "if os.path.exists(PROCESSED_IDS_FILE):\n",
                    "    with open(PROCESSED_IDS_FILE, \"r\") as f: processed_ids = set(line.strip() for line in f)\n",
                    "\n",
                    "if not os.path.exists(CACHE_DIR):\n",
                    "    print(f\"[!] Không tìm thấy thư mục Cache tại: {CACHE_DIR}\")\n",
                    "    sys.exit()\n",
                    "\n",
                    "cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]\n",
                    "print(f\"[*] Tìm thấy {len(cache_files)} bài. Đã chấm xong: {len(processed_ids)}\")\n",
                    "\n",
                    "with jsonlines.open(OUTPUT_FILE, mode='a') as writer, open(PROCESSED_IDS_FILE, \"a\") as track_file:\n",
                    "    for filename in tqdm(cache_files, desc=\"Local Judging\"):\n",
                    "        prob_id = filename.replace('.json', '')\n",
                    "        if prob_id in processed_ids: continue\n",
                    "            \n",
                    "        try:\n",
                    "            with open(os.path.join(CACHE_DIR, filename), 'r', encoding='utf-8') as f:\n",
                    "                cached_data = json.load(f)\n",
                    "            \n",
                    "            # Xử lý linh hoạt định dạng cache (List hoặc Dict)\n",
                    "            if isinstance(cached_data, list):\n",
                    "                rollouts = cached_data\n",
                    "                problem_text = \"\" # Cần nạp từ dataset nếu cần\n",
                    "            else:\n",
                    "                rollouts = cached_data.get('rollouts', [])\n",
                    "                problem_text = cached_data.get('problem', \"\")\n",
                    "            \n",
                    "            if len(rollouts) < 2: continue\n",
                    "            \n",
                    "            # Chấm điểm song song các cặp\n",
                    "            matrices = judge.evaluate_pairs(problem_text, prob_id, rollouts)\n",
                    "            \n",
                    "            if matrices:\n",
                    "                writer.write({\n",
                    "                    \"problem_id\": prob_id,\n",
                    "                    \"problem\": problem_text,\n",
                    "                    \"generated_rollouts\": rollouts,\n",
                    "                    \"distance_matrices\": matrices\n",
                    "                })\n",
                    "                track_file.write(f\"{prob_id}\\n\")\n",
                    "                track_file.flush()\n",
                    "        except Exception as e:\n",
                    "            print(f\"\\n[!] Lỗi xử lý {prob_id}: {e}\")\n",
                    "            continue\n",
                    "\n",
                    "print(\"\\n[*] HOÀN THÀNH! Dữ liệu đã sẵn sàng tại distillation_data.jsonl\")"
                ]
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4
    }

    with open("phase1_judge_only.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("[+] Đã cập nhật make2.py thành công!")

if __name__ == "__main__":
    create_judge_only_notebook()
