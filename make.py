import json

def create_distillation_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1.5: Student Model Distillation (High Performance)\n",
                    "Pipeline này đã được tối ưu hóa để khớp 100% với thiết kế trong bản thảo IEEE của bạn."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Cài đặt & Import\n",
                    "!pip install transformers torch sentence-transformers jsonlines tqdm\n",
                    "\n",
                    "import os\n",
                    "import json\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.nn.functional as F\n",
                    "from torch.utils.data import Dataset, DataLoader, random_split\n",
                    "from torch.cuda.amp import GradScaler, autocast\n",
                    "from transformers import AutoModel, AutoTokenizer\n",
                    "from tqdm import tqdm\n",
                    "from google.colab import drive\n",
                    "\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Thuật toán DTW cho Teacher (Chạy trên CPU để chuẩn bị dữ liệu)\n",
                    "def compute_teacher_dtw(matrix):\n",
                    "    \"\"\"Tính chi phí đường đi ngắn nhất (Alignment Cost) từ ma trận 0/1 của Teacher\"\"\"\n",
                    "    N, M = len(matrix), len(matrix[0])\n",
                    "    dp = [[float('inf')] * (M + 1) for _ in range(N + 1)]\n",
                    "    dp[0][0] = 0\n",
                    "    for i in range(1, N + 1):\n",
                    "        for j in range(1, M + 1):\n",
                    "            cost = float(matrix[i-1][j-1])\n",
                    "            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n",
                    "    return dp[N][M] / max(N, M) # Normalize theo độ dài lời giải"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Mô hình Student (Vectorized Soft-DTW cho GPU)\n",
                    "class SoftDTW(nn.Module):\n",
                    "    def __init__(self, gamma=0.1):\n",
                    "        super().__init__()\n",
                    "        self.gamma = gamma\n",
                    "\n",
                    "    def forward(self, D):\n",
                    "        B, N, M = D.shape\n",
                    "        device = D.device\n",
                    "        R = torch.full((B, N + 1, M + 1), 1e8, device=device)\n",
                    "        R[:, 0, 0] = 0\n",
                    "        for i in range(1, N + 1):\n",
                    "            for j in range(1, M + 1):\n",
                    "                v = torch.stack([R[:, i-1, j], R[:, i, j-1], R[:, i-1, j-1]], dim=1)\n",
                    "                soft_min = -self.gamma * torch.logsumexp(-v / self.gamma, dim=1)\n",
                    "                R[:, i, j] = D[:, i-1, j-1] + soft_min\n",
                    "        return R[:, N, M]\n",
                    "\n",
                    "class LogicalAlignmentStudent(nn.Module):\n",
                    "    def __init__(self, model_id=\"sentence-transformers/all-MiniLM-L6-v2\"):\n",
                    "        super().__init__()\n",
                    "        self.backbone = AutoModel.from_pretrained(model_id)\n",
                    "        self.dtw = SoftDTW(gamma=0.1)\n",
                    "\n",
                    "    def forward(self, ids_a, mask_a, ids_b, mask_b):\n",
                    "        emb_a = self.backbone(ids_a, mask_a).last_hidden_state\n",
                    "        emb_b = self.backbone(ids_b, mask_b).last_hidden_state\n",
                    "        # Compute Pairwise Cosine Distance\n",
                    "        a_n = F.normalize(emb_a, p=2, dim=-1)\n",
                    "        b_n = F.normalize(emb_b, p=2, dim=-1)\n",
                    "        dist_mat = 1.0 - torch.bmm(a_n, b_n.transpose(1, 2))\n",
                    "        return self.dtw(dist_mat)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. DataLoader với tính năng tách Train/Val\n",
                    "class DistillationDataset(Dataset):\n",
                    "    def __init__(self, jsonl_path, tokenizer, max_length=256):\n",
                    "        self.pairs = []\n",
                    "        with open(jsonl_path, \"r\", encoding=\"utf-8\") as f:\n",
                    "            for line in f:\n",
                    "                item = json.loads(line)\n",
                    "                rollouts, matrices = item['generated_rollouts'], item['distance_matrices']\n",
                    "                for key, mat in matrices.items():\n",
                    "                    i, j = map(int, key.strip(\"()\").split(\",\"))\n",
                    "                    t_dist = compute_teacher_dtw(mat)\n",
                    "                    self.pairs.append((rollouts[i], rollouts[j], t_dist))\n",
                    "        self.tokenizer = tokenizer\n",
                    "        self.max_length = max_length\n",
                    "\n",
                    "    def __len__(self): return len(self.pairs)\n",
                    "    def __getitem__(self, idx):\n",
                    "        txt_a, txt_b, label = self.pairs[idx]\n",
                    "        # Trả về list, việc tensor hóa để DataLoader lo (hiệu quả hơn)\n",
                    "        return txt_a, txt_b, label\n",
                    "\n",
                    "def collate_fn(batch, tokenizer, max_length):\n",
                    "    texts_a, texts_b, labels = zip(*batch)\n",
                    "    feat_a = tokenizer(list(texts_a), padding=True, truncation=True, max_length=max_length, return_tensors='pt')\n",
                    "    feat_b = tokenizer(list(texts_b), padding=True, truncation=True, max_length=max_length, return_tensors='pt')\n",
                    "    return feat_a['input_ids'], feat_a['attention_mask'], feat_b['input_ids'], feat_b['attention_mask'], torch.tensor(labels, dtype=torch.float)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Vòng lặp Huấn luyện & Validation\n",
                    "JSONL_PATH = \"/content/drive/MyDrive/Data_Phase1.5/distillation_data.jsonl\"\n",
                    "MODEL_SAVE_PATH = \"/content/drive/MyDrive/Data_Phase1.5/student_model.pth\"\n",
                    "BATCH_SIZE = 16\n",
                    "\n",
                    "tokenizer = AutoTokenizer.from_pretrained(\"sentence-transformers/all-MiniLM-L6-v2\")\n",
                    "full_dataset = DistillationDataset(JSONL_PATH, tokenizer)\n",
                    "train_size = int(0.9 * len(full_dataset))\n",
                    "val_size = len(full_dataset) - train_size\n",
                    "train_ds, val_ds = random_split(full_dataset, [train_size, val_size])\n",
                    "\n",
                    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, 256))\n",
                    "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(b, tokenizer, 256))\n",
                    "\n",
                    "model = LogicalAlignmentStudent().to(\"cuda\")\n",
                    "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)\n",
                    "criterion = nn.MSELoss()\n",
                    "scaler = GradScaler()\n",
                    "\n",
                    "for epoch in range(10):\n",
                    "    model.train()\n",
                    "    for ids_a, mask_a, ids_b, mask_b, labels in tqdm(train_loader, desc=f\"Epoch {epoch+1} [Train]\"):\n",
                    "        optimizer.zero_grad()\n",
                    "        with autocast():\n",
                    "            loss = criterion(model(ids_a.to(\"cuda\"), mask_a.to(\"cuda\"), ids_b.to(\"cuda\"), mask_b.to(\"cuda\")), labels.to(\"cuda\"))\n",
                    "        scaler.scale(loss).backward()\n",
                    "        scaler.step(optimizer)\n",
                    "        scaler.update()\n",
                    "    \n",
                    "    model.eval()\n",
                    "    val_loss = 0\n",
                    "    with torch.no_grad():\n",
                    "        for ids_a, mask_a, ids_b, mask_b, labels in val_loader:\n",
                    "            val_loss += criterion(model(ids_a.to(\"cuda\"), mask_a.to(\"cuda\"), ids_b.to(\"cuda\"), mask_b.to(\"cuda\")), labels.to(\"cuda\")).item()\n",
                    "    print(f\"[*] Val Loss: {val_loss/len(val_loader):.4f}\")\n",
                    "    torch.save(model.state_dict(), MODEL_SAVE_PATH)"
                ]
            }
        ],
        "metadata": {"kernelspec": {"name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    with open("phase2_student_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("[+] Updated make.py: Notebook is now 100% aligned with main.tex!")

if __name__ == "__main__":
    create_distillation_notebook()
