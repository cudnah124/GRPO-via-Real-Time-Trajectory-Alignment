import os
import hashlib
from datasets import load_dataset

def get_problem_id(problem_text):
    """Băm nội dung bài toán thành MD5 để theo dõi gọn nhẹ"""
    return hashlib.md5(problem_text.encode('utf-8')).hexdigest()

class MathDataset:
    @staticmethod
    def load_with_resume(processed_ids_file, cache_dir=None, sample_size=None):
        """
        Load dataset OpenThoughts-114k.
        Thứ tự: Math Filter -> Shuffle/Sample -> Processed/Cache Filter.
        """
        print("Loading OpenThoughts-114k dataset...")
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
        except Exception as e:
            print(f"Lỗi tải cấu hình 'metadata', dùng 'default': {e}")
            ds = load_dataset("open-thoughts/OpenThoughts-114k", "default", split="train")
        
        # 1. Lọc miền Toán học (Math)
        ds_math = ds.filter(lambda x: str(x.get('domain', '')).lower().strip() == 'math')
        
        # 2. Lấy mẫu trước (Sử dụng seed cố định để luôn bốc đúng tập này)
        if sample_size is not None:
            # Shuffle với seed cố định để đảm bảo tính nhất quán qua các phiên làm việc
            ds_math = ds_math.shuffle(seed=42).select(range(min(sample_size, len(ds_math))))
            print(f"Đã chọn tập mẫu cố định {len(ds_math)} bài toán.")

        # 3. Load danh sách đã xử lý
        processed_ids = set()
        if os.path.isfile(processed_ids_file):
            with open(processed_ids_file, "r") as f:
                for line in f:
                    processed_ids.add(line.strip())
            
        # 4. Lọc bỏ những bài đã làm (trong .jsonl hoặc trong cache)
        def is_not_done(example):
            pid = get_problem_id(example['problem'])
            if pid in processed_ids:
                return False
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{pid}.json")
                if os.path.exists(cache_path):
                    return False
            return True
            
        ds_final = ds_math.filter(is_not_done)
        print(f"Trong tập mẫu này, còn lại {len(ds_final)} bài chưa hoàn thành.")
            
        return ds_final, processed_ids
