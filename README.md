# CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment

Hệ thống nghiên cứu và ứng dụng đánh giá khách sạn đa khía cạnh (**Aspect-Based Sentiment Analysis - ABSA**), trích xuất câu bằng chứng (*rationale extraction*) và kiểm soát hiện tượng rò rỉ thông tin chéo (*cross-aspect leakage*) dựa trên kiến trúc **CRAM-ABSA** (*Contrastive Rationale-Aligned Multi-Task Architecture with Embedding-Based Retrieval - EBR*).

- **Mã nguồn Repository:** [https://github.com/trietnguyen2611/CRAM-ABSA-EBR-Hotel-Reviews](https://github.com/trietnguyen2611/CRAM-ABSA-EBR-Hotel-Reviews)
- **Bài báo Khoa học:** Chuẩn mẫu Springer Nature (`docs/main_paper/sn-article.tex` -> `sn-article.pdf`)

---

## 🌟 Tính năng nổi bật

1. **Bộ 6 Khía cạnh chuẩn hóa (Canonical Aspects):**
   - `Service`: Dịch vụ khách hàng, thái độ phục vụ của nhân viên, lễ tân, buồng phòng.
   - `Facility`: Cơ sở vật chất, phòng ngủ, phòng tắm, vệ sinh, cách âm, điều hòa.
   - `Experience`: Trải nghiệm lưu trú tổng thể của du khách.
   - `Amenity`: Tiện ích đi kèm (bể bơi, buffet/ẩm thực, phòng gym, spa, wifi, bãi đỗ xe).
   - `Loyalty`: Mức độ trung thành, khả năng quay lại hoặc gợi ý cho người khác.
   - `Branding`: Giá cả, đẳng cấp thương hiệu, tỷ lệ chất lượng/giá tiền (value for money).

2. **Dual-Stage Pipeline tiên tiến:**
   - **Giai đoạn 1 (Linguistic & Dense Retrieval):** Kết hợp phân tích ngữ pháp, xử lý từ phủ định (NLTK) và truy xuất đặc trưng ngữ nghĩa bằng Embedding-Based Retrieval (EBR) để trích xuất bằng chứng (rationale) theo từng khía cạnh.
   - **Giai đoạn 2 (Multi-Task BERT Backbone):** Tích hợp 20 đặc trưng dạng bảng (tabular features) qua cơ chế Residual Gating, sử dụng hàm phạt ma trận trực giao Gram (Orthogonal Gram Penalty) để loại bỏ rò rỉ ngữ nghĩa chéo (Cross-Aspect Leakage giảm xuống **0.24%**) và hồi quy thứ bậc có điều kiện (CORN - Conditional Ordinal Regression) giữ đúng tính đơn điệu của bậc thang sao đánh giá.

3. **Hiệu năng vượt trội trên tập kiểm thử (Test Split 999 reviews từ 9,990 reviews TripAdvisor):**
   - **Test Loss:** Đạt `0.2552` (giảm hơn 50% so với các baseline `0.5143 - 0.6032`).
   - **6-Aspect Macro-F1:** Đạt `72.55%` (+20.00% so với BERT-Base, +20.43% so với WSD-BiLSTM-GCN).
   - **Macro-MAE:** Đạt `0.3520`.
   - **Rationale F1:** Đạt `70.84%` (+20.18% đến +61.07% so với các mô hình tham chiếu).
   - **Acc@0:** `92.39%`.

---

## 📁 Cấu trúc thư mục

```text
├── data/
│   ├── dataset/         # Dữ liệu TripAdvisor gốc (TripAdvisor_EN.json - 44MB)
│   ├── df/              # Dữ liệu đã xử lý: hotel_reviews.parquet (42 cột, 9,990 mẫu) & chatbot.db (SQLite)
│   └── chart_img/       # 15+ biểu đồ phân tích EDA và biểu đồ kết quả đối sánh benchmark
├── train/
│   ├── train_model/     # Huấn luyện mô hình cơ sở v1 (trainv1.ipynb)
│   └── train_method/    # Huấn luyện các cơ chế mở rộng / ablation methods (trainmethod.ipynb)
├── model/
│   ├── modelv1/         # Checkpoint modelv1.pt, ebr_exemplar_bank.pkl & training_history_v1.json
│   └── method/          # Checkpoint so sánh các phương pháp (modelmethod1.pt, modelmethod9.pt,...)
├── benchmark/
│   └── benchmark.py     # Web GUI đối sánh hiệu năng, kiểm thử đơn lẻ và đo lường trực quan
├── desktop/             # Ứng dụng Desktop thương mại (Electron + Python Backend + Vanilla CSS/JS)
├── docs/
│   ├── main_paper/      # Mã nguồn bài báo LaTeX Springer Nature (sn-article.tex, build.sh, .bib)
│   ├── src_paper/       # 30 bài báo PDF nghiên cứu nền tảng (1.pdf -> 30.pdf)
│   └── Springer_Nature_LaTeX_Template/ # Bộ template mẫu gốc từ Springer Nature
├── Project.ipynb        # Pipeline tiền xử lý dữ liệu, trích xuất đặc trưng & vẽ biểu đồ EDA
└── requirements.txt     # Danh sách thư viện Python phụ thuộc
```

---

## ⚙️ Cài đặt môi trường

### 1. Cài đặt Python Virtual Environment
Yêu cầu Python **3.10 – 3.12**:

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo:
# Trên macOS / Linux:
source .venv/bin/activate
# Trên Windows:
# .venv\Scripts\activate

# Nâng cấp pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Cài đặt môi trường cho Desktop App
Yêu cầu Node.js **20+** (nếu muốn chạy hoặc đóng gói ứng dụng Desktop):

```bash
npm --prefix desktop install
```

---

## 🚀 Hướng dẫn thực thi

### 1. Xử lý dữ liệu & Phân tích Khám phá (EDA)
Mở file [Project.ipynb](Project.ipynb) trong VSCode / Jupyter Lab để chạy luồng tiền xử lý:
- Tải và phân tích dữ liệu `data/dataset/TripAdvisor_EN.json`.
- Dual-stream cleaning: Kết hợp lemmatization ngôn ngữ học và bảo toàn cấu trúc token của BERT.
- Trích xuất 20 biến đặc trưng ngữ cảnh và nhãn rating theo 6 khía cạnh.
- Xuất dữ liệu sạch ra `data/df/hotel_reviews.parquet` (9,990 bản ghi) và lưu vào cơ sở dữ liệu `data/df/chatbot.db`.
- Tự động kết xuất toàn bộ biểu đồ vào thư mục `data/chart_img/`.

### 2. Huấn luyện Mô hình
Các Jupyter Notebook trong thư mục `train/` được thiết kế có thể chạy cục bộ (GPU/MPS/CPU) hoặc trên Kaggle/Google Colab:
- **[train/train_model/trainv1.ipynb](train/train_model/trainv1.ipynb)**: Huấn luyện kiến trúc CRAM-ABSA kết hợp BERT Backbone với 6 khía cạnh chuẩn hóa (`Service`, `Facility`, `Experience`, `Amenity`, `Loyalty`, `Branding`), trích xuất Rationale và EBR.
- **[train/train_method/trainmethod.ipynb](train/train_method/trainmethod.ipynb)**: Huấn luyện các phương pháp đối chứng (Ablation study, Contrastive Loss, các biến thể trọng số).

### 3. Chạy Benchmark Web GUI
Hệ thống cung cấp một bảng điều khiển Web Dashboard trực quan để đối sánh các mô hình:

```bash
python benchmark/benchmark.py
```
- Server mở tại `http://127.0.0.1:8765`.
- Hỗ trợ nhập đánh giá tùy ý để kiểm tra kết quả phân loại 6 khía cạnh theo thời gian thực cùng đoạn trích chứng minh (rationale).
- Đánh giá toàn bộ tập test (999 review), hiển thị ma trận nhầm lẫn, biểu đồ phân phối lỗi và các chỉ số Macro-F1, Macro-MAE, Acc@0, Rationale F1.

### 4. Khởi chạy Ứng dụng Desktop
Ứng dụng hoàn chỉnh tích hợp giao diện hiện đại kết nối trực tiếp với backend AI:

```bash
# Khởi chạy ứng dụng trong chế độ Development:
npm --prefix desktop run desktop

# Đóng gói cài đặt ứng dụng:
npm --prefix desktop run dist:mac  # Xuất file DMG / ZIP (macOS)
npm --prefix desktop run dist:win  # Xuất file EXE / NSIS installer (Windows)
```

---

## 📝 Biên dịch Bài báo Khoa học (Springer Nature)

Toàn bộ tài liệu học thuật được soạn thảo theo khuôn mẫu chính thức của **Springer Nature** tại `docs/main_paper/`.

### 1. Biên dịch ra PDF
Thư mục tích hợp script tự động `build.sh` (xử lý 3-pass `pdflatex` + `bibtex`):

```bash
# Biên dịch tài liệu hoàn chỉnh:
bash docs/main_paper/build.sh

# Hoặc biên dịch file nguồn chỉ định:
bash docs/main_paper/build.sh sn-article.tex

# Dọn dẹp các tệp log / tạm (.aux, .bbl, .blg, .out, .toc,...):
bash docs/main_paper/build.sh --clean
```
File PDF hoàn thiện: `docs/main_paper/sn-article.pdf`.

### 2. Quản lý Tài liệu tham khảo
- Toàn bộ **30 bài báo tham chiếu** chuẩn hóa quốc tế được lưu trữ tương ứng trong `docs/src_paper/` (`1.pdf` đến `30.pdf`).
- Thông tin trích dẫn chi tiết nằm tại `docs/main_paper/sn-bibliography.bib` với các key trích dẫn được đánh số tương ứng (`ref01_...` tới `ref30_...`), đảm bảo thứ tự trích dẫn xuất hiện đồng bộ trong bài báo.
