import json

def create_judge_only_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1.5: Offline Judge Only (Distillation Data Preparation)\n",
                    "Notebook này dùng để chấm điểm các Rollouts đã được sinh bởi vLLM và lưu trong Cache.\n",
                    "\n",
                    "**Lưu ý:** Bạn có thể chạy notebook này trên CPU Runtime để tiết kiệm GPU quota."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Setup & Mount Drive\n",
                    "!pip install jsonlines tqdm openai\n",
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
                    "# ĐỊNH VỊ FOLDER CODE (Kiểm tra lại đường dẫn này trên Drive)\n",
                    "CODE_PATH = '/content/drive/MyDrive/conference-latex-template/Code'\n",
                    "if CODE_PATH not in sys.path:\n",
                    "    sys.path.append(CODE_PATH)\n",
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
                    "# 2. Cấu hình đường dẫn dữ liệu\n",
                    "DRIVE_BASE = \"/content/drive/MyDrive/Data_Phase1.5\"\n",
                    "CACHE_DIR = f\"{DRIVE_BASE}/rollouts_cache\"\n",
                    "OUTPUT_FILE = f\"{DRIVE_BASE}/distillation_data.jsonl\"\n",
                    "PROCESSED_IDS_FILE = f\"{DRIVE_BASE}/processed_ids_judge.txt\"\n",
                    "\n",
                    "# Khởi tạo Judge\n",
                    "judge = AlignmentJudge(model_id=config.JUDGE_MODEL_ID)\n",
                    "print(f\"[*] Sử dụng Judge Model: {config.JUDGE_MODEL_ID}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Quét Cache và tiến hành chấm điểm\n",
                    "if not os.path.exists(CACHE_DIR):\n",
                    "    raise Exception(f\"Không tìm thấy thư mục Cache tại: {CACHE_DIR}\")\n",
                    "\n",
                    "processed_ids = set()\n",
                    "if os.path.exists(PROCESSED_IDS_FILE):\n",
                    "    with open(PROCESSED_IDS_FILE, \"r\") as f:\n",
                    "        processed_ids = set(line.strip() for line in f)\n",
                    "\n",
                    "cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]\n",
                    "print(f\"[*] Tìm thấy {len(cache_files)} bài trong Cache. Đã chấm xong: {len(processed_ids)}\")\n",
                    "\n",
                    "with jsonlines.open(OUTPUT_FILE, mode='a') as writer, \\\n",
                    "     open(PROCESSED_IDS_FILE, \"a\") as track_file:\n",
                    "    \n",
                    "    for filename in tqdm(cache_files, desc=\"Judging Progress\"):\n",
                    "        prob_id = filename.replace('.json', '')\n",
                    "        if prob_id in processed_ids: continue\n",
                    "            \n",
                    "        try:\n",
                    "            with open(os.path.join(CACHE_DIR, filename), 'r', encoding='utf-8') as f:\n",
                    "                cached_data = json.load(f)\n",
                    "                problem_text = cached_data.get('problem')\n",
                    "                rollouts = cached_data.get('rollouts', [])\n",
                    "            \n",
                    "            if not problem_text or len(rollouts) < 2: continue\n",
                    "                \n",
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
                    "            print(f\"\\n[!] Lỗi tại {prob_id}: {e}\")\n",
                    "            continue\n",
                    "\n",
                    "print(\"\\n[*] Hoàn thành chấm điểm!\")"
                ]
            }
        ],
        "metadata": {"kernelspec": {"name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4
    }

    with open("phase1_judge_only.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("[+] Đã tạo file 'phase1_judge_only.ipynb' thành công!")

if __name__ == "__main__":
    create_judge_only_notebook()
