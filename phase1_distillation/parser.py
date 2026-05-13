import json
import re

def parse_distance_matrix(llm_output):
    """
    Trích xuất mảng JSON 2 chiều từ output của LLM.
    Trả về mảng 2D hoặc raise ValueError nếu không hợp lệ.
    """
    clean_text = llm_output.strip()
    
    # Thử tìm kiếm khối mảng trong ngoặc vuông lớn nhất
    match = re.search(r'\[\s*\[.*\]\s*\]', clean_text, re.DOTALL)
    if match:
        clean_text = match.group(0)
        
    try:
        matrix = json.loads(clean_text)
        
        # Validation cơ bản
        if not isinstance(matrix, list) or len(matrix) == 0:
            raise ValueError("Output không phải là mảng hoặc mảng rỗng.")
            
        for row in matrix:
            if not isinstance(row, list):
                raise ValueError("Output không phải mảng 2 chiều.")
            for val in row:
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Phần tử không phải là số: {val}")
                    
        return matrix
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi cú pháp JSON: {e}")
