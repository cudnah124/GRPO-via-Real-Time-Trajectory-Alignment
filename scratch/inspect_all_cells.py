import json
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    notebook_path = "phase2_student_training.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for i, cell in enumerate(data.get("cells", [])):
        cell_type = cell.get("cell_type", "")
        source_text = "".join(cell.get("source", []))
        print(f"=== Cell {i} ({cell_type}) ===")
        print(source_text[:300].replace('\n', ' '))
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
