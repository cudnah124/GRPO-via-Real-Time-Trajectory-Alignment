import json

def create_judge_only_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1.5: Offline Local Judge (High Speed - Problem Batching)\n",
                    "Notebook này sử dụng model **Qwen2.5-7B-AWQ** để chấm điểm song song nhiều bài toán.\n",
                    "\n",
                    "**Tối ưu:** Gom 20 bài (120 cặp) vào 1 lượt xử lý GPU."
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
                    "# 3. Tiến hành chấm điểm hàng loạt (PROBLEM BATCHING)\n",
                    "PROBLEM_BATCH_SIZE = 20 # Gom 20 bài (khoảng 120 cặp) vào 1 lượt xử lý GPU\n",
                    "\n",
                    "processed_ids = set()\n",
                    "if os.path.exists(PROCESSED_IDS_FILE):\n",
                    "    with open(PROCESSED_IDS_FILE, \"r\") as f: processed_ids = set(line.strip() for line in f)\n",
                    "\n",
                    "if not os.path.exists(CACHE_DIR):\n",
                    "    print(f\"[!] Không tìm thấy thư mục Cache tại: {CACHE_DIR}\")\n",
                    "    sys.exit()\n",
                    "\n",
                    "# Chỉ lấy các file rollouts, bỏ qua các file judge_*.json\n",
                    "cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json') and not f.startswith('judge_')]\n",
                    "pending_files = [f for f in cache_files if f.replace('.json', '') not in processed_ids]\n",
                    "\n",
                    "print(f\"[*] Tìm thấy {len(cache_files)} bài. Cần chấm: {len(pending_files)}\")\n",
                    "\n",
                    "with jsonlines.open(OUTPUT_FILE, mode='a') as writer, open(PROCESSED_IDS_FILE, \"a\") as track_file:\n",
                    "    # Vòng lặp theo Batch bài toán\n",
                    "    for i in tqdm(range(0, len(pending_files), PROBLEM_BATCH_SIZE), desc=\"Batch Judging\"):\n",
                    "        batch_filenames = pending_files[i : i + PROBLEM_BATCH_SIZE]\n",
                    "        batch_data = []\n",
                    "        \n",
                    "        # Load dữ liệu của cả batch\n",
                    "        for filename in batch_filenames:\n",
                    "            prob_id = filename.replace('.json', '')\n",
                    "            try:\n",
                    "                with open(os.path.join(CACHE_DIR, filename), 'r', encoding='utf-8') as f:\n",
                    "                    cached_data = json.load(f)\n",
                    "                \n",
                    "                if isinstance(cached_data, list):\n",
                    "                    batch_data.append({\"id\": prob_id, \"problem\": \"\", \"rollouts\": cached_data})\n",
                    "                else:\n",
                    "                    batch_data.append({\"id\": prob_id, \"problem\": cached_data.get('problem', \"\"), \"rollouts\": cached_data.get('rollouts', [])})\n",
                    "            except:\n",
                    "                continue\n",
                    "        \n",
                    "        if not batch_data: continue\n",
                    "        \n",
                    "        # CHẤM ĐIỂM SONG SONG TOÀN BỘ BATCH (Cực nhanh)\n",
                    "        batch_results = judge.evaluate_batch(batch_data, cache_dir=CACHE_DIR)\n",
                    "        \n",
                    "        # Ghi kết quả vào file cuối\n",
                    "        for data in batch_data:\n",
                    "            p_id = data['id']\n",
                    "            matrices = batch_results.get(p_id, {})\n",
                    "            if matrices:\n",
                    "                writer.write({\n",
                    "                    \"problem_id\": p_id,\n",
                    "                    \"problem\": data['problem'],\n",
                    "                    \"generated_rollouts\": data['rollouts'],\n",
                    "                    \"distance_matrices\": matrices\n",
                    "                })\n",
                    "                track_file.write(f\"{p_id}\\n\")\n",
                    "                track_file.flush()\n",
                    "\n",
                    "print(\"\\n[*] HOÀN THÀNH XÉ GIÓ! Dữ liệu đã sẵn sàng.\")"
                ]
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4
    }

    with open("phase1_judge_only.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("[+] Đã cập nhật make2.py thành công với PROBLEM BATCHING!")

if __name__ == "__main__":
    create_judge_only_notebook()
