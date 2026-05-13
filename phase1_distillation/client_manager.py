import os
import itertools
from openai import OpenAI
import phase1_distillation.config as config

class TokenRotator:
    def __init__(self):
        self.index = 0

    def get_client(self):
        """
        Lấy client với token tiếp theo. 
        Truy xuất trực tiếp từ config.HF_TOKENS để tránh lỗi cache module.
        """
        # Load tokens từ config (đã được reload trong notebook nếu cần)
        tokens = getattr(config, 'HF_TOKENS', [])
        
        if not tokens:
            # Fallback về biến môi trường đơn lẻ
            env_token = os.getenv('HF_TOKEN')
            if env_token:
                tokens = [env_token]
            else:
                tokens = ["no_token_provided"]
        
        # Xoay vòng index
        token = tokens[self.index % len(tokens)]
        self.index += 1
        
        return OpenAI(
            base_url=config.API_BASE_URL,
            api_key=token
        )

# Singleton instance
rotator = TokenRotator()
