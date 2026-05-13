import os

# Cấu hình cho OpenAI Client gọi tới Hugging Face Router
API_BASE_URL = "https://router.huggingface.co/v1"

# Bảo mật: Lấy từ biến môi trường HF_TOKEN
HF_TOKEN = os.getenv('HF_TOKEN', '')

# Model name (User gợi ý dùng thêm hậu tố :novita nếu cần)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

K_ROLLOUTS = 4
MAX_RETRIES = 3

DRIVE_OUTPUT_FILE = "/content/drive/MyDrive/Data_Phase1/phase1_generated_rollouts.jsonl"
DRIVE_PROCESSED_IDS = "/content/drive/MyDrive/Data_Phase1/processed_ids.txt"
