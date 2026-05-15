import json
with open(r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\phase1_data_generation.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "generator = MathRolloutGenerator(model_id=config.MODEL_ID)" in line:
                new_source.append(line.replace("config.MODEL_ID", "config.GENERATOR_MODEL_ID"))
            elif "judge = AlignmentJudge(model_id=config.MODEL_ID)" in line:
                new_source.append(line.replace("config.MODEL_ID", "config.JUDGE_MODEL_ID"))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(r"c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\NCKH\SELF\conference-latex-template\Code\phase1_data_generation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
