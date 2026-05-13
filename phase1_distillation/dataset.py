import os
import hashlib
from datasets import load_dataset

def get_problem_id(problem_text):
    """Băm nội dung bài toán thành MD5 để theo dõi gọn nhẹ"""
    return hashlib.md5(problem_text.encode('utf-8')).hexdigest()

class MathDataset:
    @staticmethod
    def load_with_resume(processed_ids_file, sample_size=None):
        """
        Load dataset OpenThoughts-114k và lọc bỏ những câu hỏi đã xử lý
        """
        print("Loading OpenThoughts-114k dataset...")
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
        except Exception as e:
            print(f"Lỗi tải cấu hình 'metadata', dùng 'default': {e}")
            ds = load_dataset("open-thoughts/OpenThoughts-114k", "default", split="train")
        
        # Filter math domain
        ds_math = ds.filter(lambda x: str(x.get('domain', '')).lower().strip() == 'math')
        
        # Load processed IDs
        processed_ids = set()
        if os.path.exists(processed_ids_file):
            with open(processed_ids_file, "r") as f:
                for line in f:
                    processed_ids.add(line.strip())
            print(f"Đã tìm thấy {len(processed_ids)} bài toán đã được xử lý từ trước.")
            
        # Filter out processed
        def is_not_processed(example):
            pid = get_problem_id(example['problem'])
            return pid not in processed_ids
            
        ds_filtered = ds_math.filter(is_not_processed)
        print(f"Còn lại {len(ds_filtered)} bài toán sau khi lọc.")
        
        if sample_size is not None:
            ds_filtered = ds_filtered.shuffle(seed=42).select(range(min(sample_size, len(ds_filtered))))
            print(f"Lấy mẫu ngẫu nhiên {len(ds_filtered)} bài toán để chạy.")
            
        return ds_filtered, processed_ids
