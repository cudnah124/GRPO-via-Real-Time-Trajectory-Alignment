import os
import itertools
from openai import OpenAI
import phase1_distillation.config as config

class TokenRotator:
    def __init__(self):
        # Khởi tạo iterator xoay vòng từ danh sách tokens trong config
        tokens = config.HF_TOKENS
        # Nếu rỗng thì thử lấy từ env đơn lẻ
        if not tokens or tokens == ["hf_token_1_here"]:
            env_token = os.getenv('HF_TOKEN')
            if env_token:
                tokens = [env_token]
            else:
                tokens = ["no_token_provided"]
        
        self.tokens = itertools.cycle(tokens)

    def get_client(self):
        """Trả về một OpenAI client mới với token tiếp theo trong danh sách"""
        token = next(self.tokens)
        return OpenAI(
            base_url=config.API_BASE_URL,
            api_key=token
        )

# Singleton instance để dùng chung trong toàn bộ project
rotator = TokenRotator()
