import os

# Bạn có thể dùng URL của OpenRouter hoặc Hugging Face tuỳ ý
API_URL = os.getenv("API_URL", "https://openrouter.ai/api/v1/chat/completions")
API_KEY = os.getenv("API_KEY", "your_api_key_here")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" # Dùng chung cho cả Generator và Judge

K_ROLLOUTS = 4
MAX_RETRIES = 3

DRIVE_OUTPUT_FILE = "/content/drive/MyDrive/Data_Phase1/phase1_generated_rollouts.jsonl"
DRIVE_PROCESSED_IDS = "/content/drive/MyDrive/Data_Phase1/processed_ids.txt"
