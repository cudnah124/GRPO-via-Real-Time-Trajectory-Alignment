import json

with open(r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\phase1_data_generation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Tìm ô cài đặt pip và thêm transformers, accelerate
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any("!pip install" in line for line in cell['source']):
        # Cập nhật dòng cài đặt
        new_source = []
        for line in cell['source']:
            if "!pip install" in line:
                new_source.append("!pip install -q datasets jsonlines tqdm openai transformers accelerate\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

with open(r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\phase1_data_generation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
