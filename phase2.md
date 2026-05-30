# 1. Cài đặt & Import
!pip install transformers torch sentence-transformers jsonlines tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive')

# 2. Thuật toán DTW cho Teacher (Chạy trên CPU để chuẩn bị dữ liệu)
def compute_teacher_dtw(matrix):
    """Tính chi phí đường đi ngắn nhất (Alignment Cost) từ ma trận 0/1 của Teacher"""
    N, M = len(matrix), len(matrix[0])
    dp = [[float('inf')] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = float(matrix[i-1][j-1])
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[N][M] / max(N, M) # Normalize theo độ dài lời giải

    # 3. Mô hình Student (Vectorized Soft-DTW cho GPU)
class SoftDTW(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, D):
        B, N, M = D.shape
        device = D.device
        R = torch.full((B, N + 1, M + 1), 1e8, device=device)
        R[:, 0, 0] = 0
        for k in range(2, N + M + 1):
            i_start = max(1, k - M)
            i_end = min(N, k - 1)
            I = torch.arange(i_start, i_end + 1, device=device)
            J = k - I
            v = torch.stack([R[:, I - 1, J], R[:, I, J - 1], R[:, I - 1, J - 1]], dim=1)
            soft_min = -self.gamma * torch.logsumexp(-v / self.gamma, dim=1)
            R[:, I, J] = D[:, I - 1, J - 1] + soft_min
        return R[:, N, M]

class LogicalAlignmentStudent(nn.Module):
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.dtw = SoftDTW(gamma=0.1)

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        emb_a = self.backbone(ids_a, mask_a).last_hidden_state
        emb_b = self.backbone(ids_b, mask_b).last_hidden_state
        # Compute Pairwise Cosine Distance
        a_n = F.normalize(emb_a, p=2, dim=-1)
        b_n = F.normalize(emb_b, p=2, dim=-1)
        dist_mat = 1.0 - torch.bmm(a_n, b_n.transpose(1, 2))
        return self.dtw(dist_mat)


# 4. DataLoader với tính năng tách Train/Val
class DistillationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.pairs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                rollouts, matrices = item['generated_rollouts'], item['distance_matrices']
                for key, mat in matrices.items():
                    if mat is None or not isinstance(mat, list) or len(mat) == 0 or not isinstance(mat[0], list) or len(mat[0]) == 0:
                        continue
                    i, j = map(int, key.strip("()").split(","))
                    t_dist = compute_teacher_dtw(mat)
                    self.pairs.append((rollouts[i], rollouts[j], t_dist))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        txt_a, txt_b, label = self.pairs[idx]
        # Trả về list, việc tensor hóa để DataLoader lo (hiệu quả hơn)
        return txt_a, txt_b, label

def collate_fn(batch, tokenizer, max_length):
    texts_a, texts_b, labels = zip(*batch)
    feat_a = tokenizer(list(texts_a), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    feat_b = tokenizer(list(texts_b), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return feat_a['input_ids'], feat_a['attention_mask'], feat_b['input_ids'], feat_b['attention_mask'], torch.tensor(labels, dtype=torch.float)



# 5. Vòng lặp Huấn luyện & Validation
JSONL_PATH = "/content/drive/MyDrive/Data_Phase1.5/distillation_data.jsonl"
MODEL_SAVE_PATH = "/content/drive/MyDrive/Data_Phase1.5/student_model.pth"
BATCH_SIZE = 128

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
full_dataset = DistillationDataset(JSONL_PATH, tokenizer)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=lambda b: collate_fn(b, tokenizer, 256))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, collate_fn=lambda b: collate_fn(b, tokenizer, 256))

model = LogicalAlignmentStudent().to("cuda")
if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
        print('[*] Đã kích hoạt torch.compile() giúp tối ưu hóa GPU!')
    except Exception as e:
        print(f'[!] torch.compile() thất bại: {e}. Chạy bình thường.')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()
scaler = GradScaler()

for epoch in range(10):
    model.train()
    for ids_a, mask_a, ids_b, mask_b, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            loss = criterion(model(ids_a.to("cuda"), mask_a.to("cuda"), ids_b.to("cuda"), mask_b.to("cuda")), labels.to("cuda"))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for ids_a, mask_a, ids_b, mask_b, labels in val_loader:
            val_loss += criterion(model(ids_a.to("cuda"), mask_a.to("cuda"), ids_b.to("cuda"), mask_b.to("cuda")), labels.to("cuda")).item()
    print(f"[*] Val Loss: {val_loss/len(val_loader):.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)