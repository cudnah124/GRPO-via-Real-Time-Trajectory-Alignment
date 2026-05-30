# Phase 2: Logical Alignment - Student Model Architecture Design

Tài liệu này mô tả chi tiết thiết kế kiến trúc, chiến lược dữ liệu và phương pháp huấn luyện cho Phase 2 (Student Model) của dự án. Thiết kế này được đúc kết sau khi khắc phục các rào cản về Token-level DTW và Semantic Bias, đảm bảo mô hình có khả năng nhận diện các lỗi logic toán học tinh vi ở cấp độ ký hiệu (Symbolic-level).

---

## 1. Chiến lược Dữ liệu (Data Augmentation & Triplet Construction)

### Vấn đề cốt lõi
Các mô hình nhúng (Embedding Models) hiện nay thường mắc lỗi "Semantic Bias" (Thiên lệch ngữ nghĩa): Chúng đánh giá hai câu `(x - 2)(x - 3) = 0` và `(x + 2)(x + 3) = 0` là giống nhau 99% vì từ vựng trùng lặp gần như hoàn toàn. Để ép mô hình học logic, ta cần tạo ra các **Hard Negatives** (Mẫu sai rất khó).

### Giải pháp Augmentation
Sử dụng script `augment_dataset.py` để tiêm (inject) các lỗi logic chí mạng vào các Rollouts đúng có sẵn. Trái ngược với việc chỉ augment 50% ngẫu nhiên ban đầu, hệ thống mới sẽ đảm bảo **tỷ lệ Pos:Neg $\approx$ 1:1** thông qua chiến lược phủ sóng 100% (Coverage 100%): Với $\approx 37,000$ Positives (Cost $\approx 0.0$), ta sẽ augment toàn bộ để sinh ra chính xác 1 Negative cho mỗi Positive. Mức độ nghiêm trọng của Negative (Severity) sẽ được xoay vòng đều để đảm bảo phân phối lý tưởng:
- Cost $\approx 0.3$ (Mức Nhẹ): $\approx 12,000$ mẫu
- Cost $\approx 0.4$ (Mức Trung bình): $\approx 12,000$ mẫu
- Cost $\approx 0.5$ (Mức Nghiêm trọng): $\approx 12,000$ mẫu
*(Lưu ý: Khoảng trống `[0.1-0.3]` được phép rỗng vì thiết kế không định nghĩa "lỗi rất nhẹ").*

Áp dụng 13 quy tắc Mutation chuyên biệt được thiết kế để nhắm vào điểm yếu cốt lõi của toán học:

1. **Mutate Sign (0.5)**: Đảo ngược dấu (`+` $\leftrightarrow$ `-`, `*` $\leftrightarrow$ `/`).
2. **Mutate Keyword (0.5)**: Đảo các từ khóa logic cốt lõi.
3. **Mutate Negation (0.5)**: (Thay thế Mutate Premise) Thêm từ phủ định "not" hoặc "Assume the contrary that" vào điều kiện đầu bài.
4. **Mutate Quantifier (0.5)**: Đảo lượng từ logic (`\forall` $\leftrightarrow$ `\exists`).
5. **Mutate Deletion (0.4)**: Xóa ngẫu nhiên một bước lập luận quan trọng.
6. **Mutate Scrambling (0.4)**: (Đơn giản hóa) Dùng bất kỳ delimiter nào có trong data (`\n\nStep`, `\n-`, `\n*`, `.\n`) để chia cắt và xáo trộn. Fallback: chia đôi text và swap.
7. **Mutate Notation (0.4)**: Bẫy ký hiệu toán học tinh vi. Đã mở rộng kho từ vựng: $\leq \leftrightarrow \geq$, $\subset \leftrightarrow \supset$, $\cup \leftrightarrow \cap$, $P(A|B) \leftrightarrow P(A/B)$.
8. **Mutate Shortcut (0.5)**: (Thay thế Reasoning Trap) Xóa toàn bộ các bước trung gian, chỉ giữ lại đề bài và kết luận (skip the proof).
9. **Mutate Number (0.3)**: (Thay thế Mutate Unit) Dùng Regex quét hằng số bất kỳ và thay đổi nhẹ (+1, -1, $\times 2$). Dễ implement và luôn thành công.
10. **Mutate Circular (0.3)**: (Đơn giản hóa) Copy nguyên câu kết luận cuối cùng ném lên làm câu mở đầu.
11. **Mutate Proof Direction (0.4)**: (Đơn giản hóa) Dùng Regex cực nhanh để tìm `[Ii]f ... then ...` và đảo ngược thành `if B then A`.
12. **Mutate Conclusion Swap (0.3)**: Đánh tráo kết luận (Hoán đổi kết luận của hai bài toán khác nhau cùng domain).
13. **Mutate Noise (0.3)**: Chèn ngẫu nhiên định lý lạc đề vào phần **giữa** văn bản và xác nhận bằng assert length.
14. **Mutate Fatal Logic (0.5)**: Lỗi logic chí mạng (Universal Fallback). Chèn mệnh đề sai hoàn toàn *"However, since 1 = 0, all conditions are trivially satisfied."* vào đầu bài. Rule này đảm bảo Mức 0.5 luôn có đủ sample kể cả khi các rule khác thất bại.

### Stratified Sampling, Labeling và Logging
1. **Stratified Sampling**: Thay vì chọn ngẫu nhiên 1 rule trong 13 rules (dễ dẫn đến mất cân bằng do các rule nhẹ hay thất bại silently), thuật toán nhận một target severity (vd: 0.4), xáo trộn (shuffle) tất cả các rules thuộc mức 0.4 và thử tuần tự. Rule nào thành công đầu tiên sẽ được chọn (break), đảm bảo phân phối không bị vỡ.
2. **Hệ thống Logging**: Mọi hành vi áp dụng rule đều được track `success`/`fail`. In ra log cuối chương trình để debug các rule "chết".
3. **Gán nhãn Ma trận (Labeling)**: Các Rollout đột biến sẽ được gán cứng (hard-code) mức độ lỗi tùy thuộc vào độ nghiêm trọng của Mutation Rule. Mức Hardcode này dùng làm Negative Target:
   - Mức `0.5` (Nghiêm trọng nhất): Mutate Sign, Mutate Quantifier, Mutate Shortcut, Mutate Keyword, Mutate Negation, Mutate Fatal Logic.
   - Mức `0.4` (Trung bình): Mutate Deletion, Mutate Proof Direction, Mutate Notation, Mutate Scrambling.
   - Mức `0.3` (Nhẹ): Mutate Noise, Mutate Number, Mutate Circular, Mutate Conclusion Swap.
4. **Khởi tạo Triplet**: Trong quá trình huấn luyện bằng `TripletDataset`, hệ thống tự động nhóm dữ liệu thành các bộ 3 (Triplet):
   - **Khoảng cách Margin an toàn**: Khoảng cách an toàn giữa nhóm Positive và Negative được nới rộng đáng kể: Positive Threshold $\le 0.10$ và Negative Threshold $\ge 0.30$. Các nhãn xám ở giữa bị bỏ qua để tránh mô hình bị nhiễu.
   - **Bảo toàn Anchor**: Sử dụng `hashlib.md5` để nhóm Anchor một cách tuyệt đối an toàn. Đồng thời, loại bỏ hoàn toàn chiến lược Global Negative Fallback: Một Anchor bắt buộc phải có Local Negative để tạo Triplet. Nếu không có, Triplet đó sẽ bị hủy để duy trì cấu trúc logic nghiêm ngặt.
   - **Anchor (A)**: Một Rollout gốc chuẩn mực.
   - **Positive ($B_{pos}$)**: Gồm 2 loại được trộn có chủ đích:
     * **Loại A (Diversity Tolerance)**: Rollout đúng khác của cùng bài toán.
     * **Loại B (Anchor Calibration)**: Chính Anchor đi qua augmentation nhẹ như paraphrase.
   - **Hard Negative ($B_{neg}$)**: Rollout đột biến chứa lỗi sai (Học theo mức Cost hard-code $\in \{0.3, 0.4, 0.5\}$).
Điều này thiết lập một mục tiêu huấn luyện hình học cực kỳ nghiêm ngặt: Mô hình bị ép buộc phải nhận ra lỗi sai nhỏ nhất và đẩy không gian vector của $B_{neg}$ ra xa khỏi Anchor ít nhất một khoảng Margin.

---

## 2. Thiết kế Kiến trúc Student Model (ESIM-style Deep Interaction)

### 2.1. Backbone: Chuyển đổi hoàn toàn sang `witiko/mathberta`
* **Lý do loại bỏ `CodeBERT`**: Dù CodeBERT (Pre-train trên code) nhạy cảm với phép toán hơn MPNet, nó vẫn lạ lẫm với các mã LaTeX toán học thuần túy có trong tập dữ liệu MATH500.
* **Quyết định**: Sử dụng `witiko/mathberta` (được pre-train trực tiếp trên các paper toán học và mã nguồn LaTeX) làm backbone mặc định. Điều này loại bỏ hoàn toàn nhu cầu phải tạo corpus Warm-up riêng cho CodeBERT.

### 2.2. Vượt qua giới hạn của MaxSim & Lexical Overlap
Trong thiết kế ColBERT truyền thống, độ lệch được tính qua hàm `MaxSim` (tìm token giống nhất bằng Cosine Similarity). Tuy nhiên, vì CodeBERT là một mô hình ngôn ngữ, không gian vector của nó thiên vị nặng nề về **Lexical Overlap (Sự trùng lặp từ vựng)**. Các ký hiệu đối lập (`+` và `-`) thường có vector rất giống nhau vì chúng cùng xuất hiện trong một ngữ cảnh.
Nếu dùng `MaxSim`, Gradient sẽ bị phân tán (Whack-a-Mole problem), khiến mô hình thất bại trong việc học lỗi sai tinh vi và thoái hóa thành "máy đếm từ vựng". 

### 2.3. Kiến trúc ESIM-style (Late Deep Interaction)
Để trị tận gốc vấn đề trên, kiến trúc phân tách việc "Tìm kiếm" và "Chấm điểm" thành 2 bước, kết hợp ưu điểm của Bi-Encoder (O(N) inference) và Cross-attention (Deep Interaction). Cụ thể, backbone nặng (Transformer 12 layers) sẽ chạy độc lập trên từng câu (tốn $O(N)$). Chỉ ở bước cuối cùng, ma trận token mới thực hiện Cross-attention (tốn $O(N^2)$ nhưng rất nhẹ vì chỉ tính dot-product):
1. **Softmax Alignment**: Mô hình tính toán ma trận Attention giữa mọi cặp token của Câu A và Câu B. Sau đó dùng hàm Softmax để kéo các token tương ứng của B đắp vào A, tạo thành biểu diễn `B_aligned`. 
   *(Lưu ý 1: Để tránh bão hòa Softmax (Vanishing Gradient), ma trận Attention thô bắt buộc phải được chia cho hệ số tỷ lệ $\sqrt{H}$ (Scaled Dot-Product) trước khi đưa qua Softmax).*
   *(Lưu ý 2: Để tránh mâu thuẫn giữa Local Sliding-window và Global Cross-attention, hệ thống sẽ truncate cứng ở `max_length = 512`. Nếu rollout dài hơn, áp dụng kỹ thuật **Head-and-Tail Truncation**: Cắt 256 token đầu (Premise/Giả thiết) ghép với 256 token cuối (Conclusion/Kết luận), bỏ phần giữa).*
2. **Trích xuất đặc trưng tương phản (Contradiction Features)**: Với mỗi token của câu A, ta ghép 4 luồng thông tin: 
   $$ m_a = [\text{A}, \text{B\_aligned}, \text{A} - \text{B\_aligned}, \text{A} * \text{B\_aligned}] $$
   Phép trừ $\text{A} - \text{B\_aligned}$ đóng vai trò như một máy dò lỗi. Nếu hai token có logic nghịch đảo (vd: `+` và `-`), phép trừ này sẽ làm kích hoạt vùng nhiễu cực mạnh.
3. **Chấm điểm $d_t$ (ESIM Head)**: Các đặc trưng $m_a$ (kích thước $4H$) được đưa qua một MLP Head để xuất ra một scalar $d_t$ cho mỗi token: `Head = nn.Sequential(Linear(4H, 128), ReLU(), Linear(128, 1), Softplus(beta=10))`.
   *(Lưu ý: Việc dùng `Softplus` ở lớp cuối thay thế cho `ReLU` hoặc `Sigmoid` giúp tránh hội chứng "Gradient Blocking". Nếu mô hình dự đoán $0.0$ mà target là $>0$, Gradient âm đổ về sẽ bị ReLU chặn đứng vĩnh viễn ở vùng $\le 0$. Softplus tiệm cận 0 nhưng đạo hàm không bao giờ bằng 0, đảm bảo luồng gradient lưu thông liên tục).* Không có projection layer nào được gắn thêm sau bước Top-K Pooling.

### 2.4. Dự đoán Per-token Divergence $d_t$ và Top-K Pooling
* **Dynamic Top-K Mean Pooling Per-Sample**: Thay vì lấy số token cố định cho cả batch, thuật toán Pooling được Vectorize hóa để tính Top-K dựa trên độ dài thực tế của từng câu riêng biệt: $K_i = \max(3, \lfloor 0.05 \times \text{len}_i \rfloor)$. Hệ thống sử dụng `torch.sort` kết hợp với Tensor Mask Ranking `ranks < K_i` và ép kiểu `Float` ở bước chia trung bình để đảm bảo tính an toàn bộ nhớ và tốc độ tuyệt đối trên GPU.
* **Mean Aggregation**: Khoảng cách cuối cùng giữa 2 văn bản được tổng hợp bằng hàm trung bình cộng $0.5 \times cost_a + 0.5 \times cost_b$ (thay vì dùng `torch.max`). Việc này giúp gradient chảy đều qua cả hai bộ lọc (cho cả câu A và câu B), tránh tình trạng câu dài luôn chiếm sóng do hàm max không tham số.

---

## 3. Phương pháp Huấn luyện (Training Objective)

### 3.1. Loss Function (Multi-Task Learning)
Hàm Loss tổng quát là sự kết hợp của 2 mục tiêu:
$$ \mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_{epoch} \cdot \mathcal{L}_{Triplet} $$
*(Với siêu tham số $\lambda$ được warm-up tuyến tính mượt mà để tránh gradient shock:*
$$ \lambda_{epoch} = \begin{cases} 0.1 & \text{nếu } epoch \le 2 \\ \min\left(1.0,\ 0.1 + \frac{epoch - 2}{N_{warmup}} \times 0.9\right) & \text{nếu } epoch > 2 \end{cases} $$
*)*

1. **Absolute Calibration (MSE Loss)**: Ép mô hình dự đoán chính xác giá trị khoảng cách (scalar). Đối với Positive Targets, Teacher sử dụng thuật toán quy hoạch động (Dynamic Programming) để tìm ra Scalar Path Cost ngắn nhất, sau đó **chuẩn hóa (Normalize) về dải $[0, 1]$** bằng cách chia Path Cost cho $N + M - 1$ (chiều dài tối đa lý thuyết của path) trước khi làm target cho MSE. Negative Targets học theo Hardcoded Costs.
2. **Relative Ranking (Triplet Margin Loss)**: Đây là trái tim của quá trình học logic.
   $$ \mathcal{L}_{Triplet} = \max(0, \text{cost}_{pos} - \text{cost}_{neg} + m) $$
   Trong đó, margin được hạ xuống **$m = 0.2$**. Lý do: Vì MSE Loss đã kéo $\text{cost}_{neg}$ về mức $0.3 - 0.5$, nếu $m$ quá lớn ($0.5$), Triplet và MSE sẽ "đánh nhau" (MSE bắt bằng 0.3 nhưng Triplet bắt phải > 0.5), dẫn đến Loss không bao giờ hội tụ về 0. Với $m=0.2$, Triplet đóng vai trò là "biên an toàn", chỉ kích hoạt khi mô hình rank sai, giúp gradient ổn định.

### 3.2. Evaluation Protocol
Quá trình huấn luyện không chỉ đánh giá thông qua Loss mà còn tích hợp các metrics kiểm tra chất lượng logic alignment:
* **Kendall's $\tau$ Correlation trên Toàn tập Validation**: Quá trình đánh giá được thực hiện qua `TensorDataset` chuẩn (thay vì TripletDataset bị chia cắt). Kendall Tau đo lường sự tương quan thứ hạng giữa Cost Scalar do Student dự đoán và Target Cost trên toàn bộ phân phối dữ liệu (gồm cả nhãn xám). Điều này phản ánh bức tranh thực tế nhất về khả năng Ranking của mô hình.
* **Triplet Accuracy**: Tỷ lệ phần trăm các triplet mà mô hình xếp hạng đúng khoảng cách ($\text{cost}_{neg} > \text{cost}_{pos} + 0.2$). Triplet được reshuffle mỗi 2 epoch để Model liên tục có Hard Negatives mới.

### 3.3. Training Optimizations & Engineering Fixes
Do giới hạn VRAM trên Colab và đặc thù của kiến trúc Late Deep Interaction, quá trình huấn luyện sử dụng các chiến lược tinh chỉnh sau:
* **Dynamic Gradient Checkpointing**: Kích hoạt đổi chi phí tính toán lấy không gian bộ nhớ. Lệnh này chỉ được bật (`enable`) bên trong hàm `train()` và tắt (`disable`) trong hàm `eval()` để đảm bảo Inference luôn chạy hết tốc lực.
* **Early Stopping & Full State Checkpointing**: Hệ thống theo dõi `best_val_loss` với Patience = 3. Bất cứ khi nào tạo ra Kỷ lục mới, toàn bộ state_dict của `Model`, `Optimizer`, và `Scheduler` đều được lưu lại. Tránh tình trạng Learning Rate bị reset về 0 khi phục hồi từ checkpoint cũ.
* **Deterministic Seeds**: Khóa cứng toàn bộ hàm ngẫu nhiên (`random`, `numpy`, `torch`, `torch.cuda`) bằng Seed = 42 để đảm bảo kết quả huấn luyện có thể tái lập tuyệt đối.
* **Gradient Accumulation & FP16**: Do không đủ VRAM để chạy Batch 32, hệ thống sử dụng Batch 8 kết hợp tích lũy 4 bước. Quá trình tính toán Attention Mask sử dụng giá trị `-1e4` thay vì `-1e9` nhằm ngăn chặn lỗi tràn số 16-bit (`c10::Half`). Quá trình backward pass dùng chuẩn `torch.cuda.amp.GradScaler()` để tương thích tuyệt đối.
* **Full Unfreeze (`freeze_backbone=False`)**: Toàn bộ 110 triệu tham số của MathBERT được mở khóa và tối ưu bằng AdamW kết hợp với Warm-up Scheduler.
* **Inference Optimization**: Hàm `tokenize_ht` (Head-and-Tail Truncation) được định nghĩa tĩnh bên ngoài vòng lặp Inference để tránh chi phí overhead do khai báo lại liên tục trên lượng lớn dữ liệu test.
