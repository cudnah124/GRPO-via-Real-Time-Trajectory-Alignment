import re
with open("make.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Remove gradient_checkpointing_enable from __init__
code = code.replace(
    "        self.backbone.gradient_checkpointing_enable() # [MỚI] Đổi compute lấy VRAM\n",
    ""
)

# 2. Fix Vectorized Top-K
topk_old = """        # 6. Dynamic Top-K Pooling cho từng câu (Dựa trên độ dài thực tế)
        real_len_a = mask_a.sum(dim=1).float()
        K_a = max(3, int(0.05 * real_len_a.min().item()))
        d_t_a_masked = d_t_a.masked_fill(mask_a == 0, 0.0)
        cost_a = torch.topk(d_t_a_masked, k=K_a, dim=1).values.mean(dim=1)
        
        real_len_b = mask_b.sum(dim=1).float()
        K_b = max(3, int(0.05 * real_len_b.min().item()))
        d_t_b_masked = d_t_b.masked_fill(mask_b == 0, 0.0)
        cost_b = torch.topk(d_t_b_masked, k=K_b, dim=1).values.mean(dim=1)
        
        # 7. Aggregation"""
topk_new = """        # 5. Dynamic Top-K Pooling cho từng câu (Dựa trên độ dài thực tế, tính per-sample)
        real_len_a = mask_a.sum(dim=1).float()
        K_a = torch.clamp((0.05 * real_len_a).int(), min=3).unsqueeze(1)
        d_t_a_masked = d_t_a.masked_fill(mask_a == 0, 0.0)
        sorted_a, _ = torch.sort(d_t_a_masked, dim=1, descending=True)
        ranks_a = torch.arange(d_t_a_masked.size(1), device=d_t_a_masked.device).unsqueeze(0).expand(d_t_a_masked.size(0), -1)
        cost_a = (sorted_a * (ranks_a < K_a)).sum(dim=1) / K_a.squeeze(1)
        
        real_len_b = mask_b.sum(dim=1).float()
        K_b = torch.clamp((0.05 * real_len_b).int(), min=3).unsqueeze(1)
        d_t_b_masked = d_t_b.masked_fill(mask_b == 0, 0.0)
        sorted_b, _ = torch.sort(d_t_b_masked, dim=1, descending=True)
        ranks_b = torch.arange(d_t_b_masked.size(1), device=d_t_b_masked.device).unsqueeze(0).expand(d_t_b_masked.size(0), -1)
        cost_b = (sorted_b * (ranks_b < K_b)).sum(dim=1) / K_b.squeeze(1)
        
        # 6. Aggregation"""
code = code.replace(topk_old, topk_new)

# 3. Fix TripletDataset Anchor Hash and global negative
triplet_old = """    def __init__(self, ids_a, mask_a, ids_b, mask_b, labels,
                 pos_threshold=0.15, neg_threshold=0.25):
        self.ids_a  = ids_a
        self.mask_a = mask_a
        self.ids_b  = ids_b
        self.mask_b = mask_b
        self.labels = labels
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        # Nhóm index theo anchor (ids_a) để bảo đảm A cố định cho mỗi triplet
        self.anchor_to_idx = {}
        for i in range(len(labels)):
            h = hash(ids_a[i].numpy().tobytes())
            if h not in self.anchor_to_idx:
                self.anchor_to_idx[h] = []
            self.anchor_to_idx[h].append(i)

        self.global_neg_idx = (labels >= neg_threshold).nonzero(as_tuple=True)[0].tolist()
        self._build_triplets()

    def _build_triplets(self):
        self.triplets = []
        for indices in self.anchor_to_idx.values():
            pos_indices = [i for i in indices if self.labels[i].item() <= self.pos_threshold]
            neg_indices = [i for i in indices if self.labels[i].item() >= self.neg_threshold]
            
            for p in pos_indices:
                valid_neg_idx = []
                cost_pos = self.labels[p].item()
                if cost_pos > 0.1:
                    valid_neg_idx = [n for n in neg_indices if self.labels[n].item() >= 0.45]
                else:
                    valid_neg_idx = neg_indices
                
                if not valid_neg_idx:
                    valid_neg_idx = self.global_neg_idx
                    
                if len(valid_neg_idx) > 0:
                    n = random.choice(valid_neg_idx)
                    self.triplets.append((p, p, n))"""
triplet_new = """    def __init__(self, ids_a, mask_a, ids_b, mask_b, labels,
                 pos_threshold=0.10, neg_threshold=0.30):
        self.ids_a  = ids_a
        self.mask_a = mask_a
        self.ids_b  = ids_b
        self.mask_b = mask_b
        self.labels = labels
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        import hashlib
        # Nhóm index theo anchor (ids_a) để bảo đảm A cố định cho mỗi triplet
        self.anchor_to_idx = {}
        for i in range(len(labels)):
            h = hashlib.md5(ids_a[i].numpy().tobytes()).hexdigest()
            if h not in self.anchor_to_idx:
                self.anchor_to_idx[h] = []
            self.anchor_to_idx[h].append(i)

        self._build_triplets()

    def _build_triplets(self):
        self.triplets = []
        for indices in self.anchor_to_idx.values():
            pos_indices = [i for i in indices if self.labels[i].item() <= self.pos_threshold]
            neg_indices = [i for i in indices if self.labels[i].item() >= self.neg_threshold]
            
            for p in pos_indices:
                valid_neg_idx = []
                cost_pos = self.labels[p].item()
                if cost_pos > 0.05:
                    valid_neg_idx = [n for n in neg_indices if self.labels[n].item() >= 0.45]
                else:
                    valid_neg_idx = neg_indices
                
                # Bỏ global fallback để bảo đảm A hoàn toàn cố định
                if len(valid_neg_idx) > 0:
                    n = random.choice(valid_neg_idx)
                    self.triplets.append((p, p, n))"""
code = code.replace(triplet_old, triplet_new)

# 4. Remove LAMBDA_TRIPLET and set seed
top_vars_old = """TRIPLET_MARGIN = 0.2    # Khoảng cách tối thiểu: cost(pos) < cost(neg) - margin
LAMBDA_TRIPLET = 1.0    # Trọng số của triplet loss so với MSE loss

tokenizer = AutoTokenizer.from_pretrained("witiko/mathberta")"""
top_vars_new = """TRIPLET_MARGIN = 0.2    # Khoảng cách tối thiểu: cost(pos) < cost(neg) - margin

import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("witiko/mathberta")"""
code = code.replace(top_vars_old, top_vars_new)

# 5. Fix Validation dataset logic and Setup logic
setup_old = """    # --- TripletDataset cho validation (Track cả MSE và Triplet) ---
    val_ids_a, val_mask_a, val_ids_b, val_mask_b, val_labels = subset_tensors(val_idx)
    val_ds = TripletDataset(val_ids_a, val_mask_a, val_ids_b, val_mask_b, val_labels)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                            num_workers=0, pin_memory=True)

    model = LogicalAlignmentStudent(freeze_backbone=False).to("cuda")
    # Tải weights đã huấn luyện trước đó nếu có để tiếp tục train
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[*] Đang tải weights đã train từ {MODEL_SAVE_PATH} để tiếp tục huấn luyện...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cuda"))
            print("[+] Tải weights thành công! Tiếp tục huấn luyện...")
        except Exception as e:
            print(f"[!] Lỗi load weight cũ: {e}")
    else:
        print("[*] Không tìm thấy weights cũ. Bắt đầu huấn luyện từ đầu.")

    optimizer     = torch.optim.AdamW(model.parameters(), lr=2e-5)
    mse_criterion = nn.MSELoss()
    scaler        = torch.amp.GradScaler('cuda')
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * 10 // ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(10):"""
setup_new = """    # --- TensorDataset cho validation (Track MSE & Kendall Tau trên toàn tập) ---
    val_ids_a, val_mask_a, val_ids_b, val_mask_b, val_labels = subset_tensors(val_idx)
    val_ds = TensorDataset(val_ids_a, val_mask_a, val_ids_b, val_mask_b, val_labels)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                            num_workers=0, pin_memory=True)

    model = LogicalAlignmentStudent(freeze_backbone=False).to("cuda")
    optimizer     = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler        = torch.cuda.amp.GradScaler()
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * 10 // ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[*] Đang tải checkpoint từ {MODEL_SAVE_PATH}...")
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cuda")
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                model.load_state_dict(checkpoint)
            print("[+] Tải checkpoint thành công! Tiếp tục huấn luyện...")
        except Exception as e:
            print(f"[!] Lỗi load checkpoint cũ: {e}")
    else:
        print("[*] Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu.")

    mse_criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(10):"""
code = code.replace(setup_old, setup_new)

# 6. Gradient Checkpointing in Train
train_start_old = """        model.train()
        total_loss, total_mse, total_triplet = 0.0, 0.0, 0.0"""
train_start_new = """        model.train()
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'gradient_checkpointing_enable'):
            model.backbone.gradient_checkpointing_enable()
        total_loss, total_mse, total_triplet = 0.0, 0.0, 0.0"""
code = code.replace(train_start_old, train_start_new)

# 7. Validation loop replacement
val_loop_old = """        # --- Validation ---
        model.eval()
        val_loss, val_mse, val_triplet = 0.0, 0.0, 0.0
        correct_triplets, total_triplets = 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                ids_a, mask_a, ids_b_pos, mask_b_pos, labels_pos, \\
                    ids_b_neg, mask_b_neg, labels_neg = [x.to('cuda') for x in batch]

                with torch.autocast(device_type='cuda'):
                    cost_pos = model(ids_a, mask_a, ids_b_pos, mask_b_pos)
                    cost_neg = model(ids_a, mask_a, ids_b_neg, mask_b_neg)
                    loss_mse = mse_criterion(cost_pos, labels_pos) + mse_criterion(cost_neg, labels_neg)
                    loss_triplet = F.relu(cost_pos - cost_neg + TRIPLET_MARGIN).mean()
                    loss = loss_mse + current_lambda * loss_triplet

                val_loss += loss.item()
                val_mse += loss_mse.item()
                val_triplet += loss_triplet.item()

                correct_triplets += (cost_neg > cost_pos + TRIPLET_MARGIN).sum().item()
                total_triplets += cost_pos.size(0)
                
                all_preds.extend(cost_pos.cpu().tolist() + cost_neg.cpu().tolist())
                all_targets.extend(labels_pos.cpu().tolist() + labels_neg.cpu().tolist())

        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        avg_val_mse = val_mse / n_val
        avg_val_triplet = val_triplet / n_val
        triplet_acc = correct_triplets / total_triplets if total_triplets > 0 else 0.0
        
        import scipy.stats as stats
        tau, _ = stats.kendalltau(all_preds, all_targets)
        
        print(f"[*] Epoch {epoch+1} | Val Total: {avg_val_loss:.4f} | MSE: {avg_val_mse:.4f} | Triplet: {avg_val_triplet:.4f} | Triplet Acc: {triplet_acc*100:.2f}% | Kendall Tau: {tau:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Luôn lưu model sau mỗi epoch
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        epoch_save_path = MODEL_SAVE_PATH.replace(
            ".pth",
            f"_epoch{epoch+1}_val{avg_val_loss:.4f}.pth"
        )
        torch.save(raw_model.state_dict(), epoch_save_path)

        # Lưu dạng HuggingFace model để dễ dàng load lại về sau
        hf_save_path = epoch_save_path.replace('.pth', '_hf')
        os.makedirs(hf_save_path, exist_ok=True)
        raw_model.backbone.save_pretrained(hf_save_path)
        tokenizer.save_pretrained(hf_save_path)
        print(f"[+] Saved model to {epoch_save_path}")"""
val_loop_new = """        # --- Validation ---
        model.eval()
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'gradient_checkpointing_disable'):
            model.backbone.gradient_checkpointing_disable()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                ids_a, mask_a, ids_b, mask_b, labels = [x.to('cuda') for x in batch]

                with torch.autocast(device_type='cuda'):
                    cost = model(ids_a, mask_a, ids_b, mask_b)
                    loss = mse_criterion(cost, labels)

                val_loss += loss.item()
                all_preds.extend(cost.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        
        import scipy.stats as stats
        tau, _ = stats.kendalltau(all_preds, all_targets)
        
        print(f"[*] Epoch {epoch+1} | Val MSE: {avg_val_loss:.4f} | Kendall Tau: {tau:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early Stopping & Best Model tracking
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, MODEL_SAVE_PATH)
            
            hf_save_path = MODEL_SAVE_PATH.replace('.pth', '_hf')
            import os
            os.makedirs(hf_save_path, exist_ok=True)
            raw_model.backbone.save_pretrained(hf_save_path)
            tokenizer.save_pretrained(hf_save_path)
            print(f"[+] Mới: Best model được lưu đè vào {MODEL_SAVE_PATH}")
        else:
            patience += 1
            print(f"[-] Val Loss không giảm (Patience: {patience}/3)")
            if patience >= 3:
                print("[*] Early Stopping kích hoạt. Dừng huấn luyện!")
                break"""
code = code.replace(val_loop_old, val_loop_new)

# 8. Load model logic in test step
test_load_old = """    print(f"[*] Đang tải weights từ {MODEL_SAVE_PATH}...")
    test_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("[+] Tải weights thành công!")"""
test_load_new = """    print(f"[*] Đang tải weights từ {MODEL_SAVE_PATH}...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    if 'model' in checkpoint:
        test_model.load_state_dict(checkpoint['model'])
    else:
        test_model.load_state_dict(checkpoint)
    print("[+] Tải weights thành công!")"""
code = code.replace(test_load_old, test_load_new)

with open("make.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Done!")
