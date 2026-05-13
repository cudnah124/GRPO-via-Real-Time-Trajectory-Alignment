import os

# Sử dụng Hugging Face Router như user cung cấp
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Dùng fallback token từ file generate_ner_hf.py
HF_TOKEN = os.getenv('HF_TOKEN', '')

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

K_ROLLOUTS = 4
MAX_RETRIES = 3

DRIVE_OUTPUT_FILE = "/content/drive/MyDrive/Data_Phase1/phase1_generated_rollouts.jsonl"
DRIVE_PROCESSED_IDS = "/content/drive/MyDrive/Data_Phase1/processed_ids.txt"
