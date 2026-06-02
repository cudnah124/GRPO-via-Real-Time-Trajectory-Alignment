import os

log_path = r"C:\Users\nhanha213\.gemini\antigravity\brain\98b34f03-c4f1-4454-a252-bd4a5d107d1d\.system_generated\logs\transcript.jsonl"
query = "phase2_student_training_best.py"

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if query in line:
                print(f"Line {idx}: {line[:200]}...")
else:
    print("Log path not found")
