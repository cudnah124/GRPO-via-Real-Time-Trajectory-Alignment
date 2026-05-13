import os

# Cấu hình cho OpenAI Client gọi tới Hugging Face Router
API_BASE_URL = "https://router.huggingface.co/v1"

# Danh sách Token để xoay vòng (Lấy từ biến môi trường để bảo mật)
# Trong Notebook hãy set: os.environ['HF_TOKENS'] = 'token1,token2,token3'
HF_TOKENS_STR = os.getenv('HF_TOKENS', '')
HF_TOKENS = [t.strip() for t in HF_TOKENS_STR.split(',') if t.strip()]

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

K_ROLLOUTS = 4
MAX_RETRIES = 3

DRIVE_OUTPUT_FILE = "/content/drive/MyDrive/Data_Phase1/phase1_generated_rollouts.jsonl"
DRIVE_PROCESSED_IDS = "/content/drive/MyDrive/Data_Phase1/processed_ids.txt"
