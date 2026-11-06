# CRAM-ABSA: Hotel Review Assessment & Rationalization

Dự án nghiên cứu và phát triển hệ thống đánh giá khách sạn đa khía cạnh (Aspect-Based Sentiment Analysis - ABSA), trích xuất bằng chứng (rationale extraction) và sinh bình luận tự động dựa trên mô hình **CRAM-ABSA** (Contrastive Rationale-Aligned Multi-Task BERT).

---

## 📁 Cấu trúc thư mục

```text
├── data/
│   ├── dataset/         # Dữ liệu TripAdvisor gốc (JSON)
│   ├── df/              # Dữ liệu xử lý định dạng Parquet & SQLite database
│   └── chart_img/       # Biểu đồ phân tích dữ liệu (EDA) phục vụ bài báo
├── train/
│   ├── train_model/     # Notebook huấn luyện mô hình cơ sở (trainv1.ipynb)
│   └── train_method/    # Notebook huấn luyện các phương pháp mở rộng (trainmethod.ipynb)
├── model/               # Lưu trữ checkpoint trọng số mô hình (.pt)
├── benchmark/
│   └── benchmark.py     # Web GUI đối sánh, đo lường độ chính xác và kiểm thử
├── desktop/             # Ứng dụng Desktop hoàn chỉnh (Electron + Python backend)
├── docs/
│   ├── main_paper/      # Nguồn bài báo LaTeX Springer Nature (sn-article.tex, build.sh)
│   ├── src_paper/       # 32 tài liệu PDF tham khảo gốc (1.pdf -> 32.pdf)
│   └── Research_Plan.md # Đề cương và câu hỏi nghiên cứu (RQ1, RQ2, RQ3)
├── Project.ipynb        # Pipeline phân tích, làm sạch dữ liệu & tạo đồ thị
└── requirements.txt     # Danh sách thư viện Python phụ thuộc
```

---

## ⚙️ Cài đặt môi trường

### 1. Môi trường Python
Yêu cầu Python **3.10 – 3.12**:

```bash
# Khởi tạo và kích hoạt virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Môi trường Desktop (Node.js)
Yêu cầu Node.js **20+** (nếu chạy ứng dụng Desktop):

```bash
npm --prefix desktop install
```

---

## 🚀 Hướng dẫn chạy Code

### 1. Xử lý dữ liệu & Khám phá (EDA)
Mở và chạy file [Project.ipynb](Project.ipynb):
- Đọc dữ liệu `data/dataset/TripAdvisor_EN.json`.
- Làm sạch 2 luồng (Dual-stream cleaning: NLTK Lemmatization & BERT token preservation).
- Xuất dữ liệu sang `data/df/hotel_reviews.parquet` và tạo bộ biểu đồ phân tích trong `data/chart_img/`.

### 2. Huấn luyện mô hình
Chạy tuần tự các Jupyter Notebook trong thư mục `train/`:
- **[train/train_model/trainv1.ipynb](train/train_model/trainv1.ipynb)**: Huấn luyện kiến trúc BERT backbone cho 6 khía cạnh (Value, Rooms, Location, Cleanliness, Front Desk, Service).
- **[train/train_method/trainmethod.ipynb](train/train_method/trainmethod.ipynb)**: Huấn luyện cơ chế đối sánh Contrastive Loss và Rationale Alignment.

> **Lưu ý:** Tuyệt đối không commit file trọng số lớn (`.pt`, `.bin`, `.safetensors`) lên Git. Khi huấn luyện trên Colab/Kaggle, lưu trọng số vào thư mục tạm `/kaggle/working/` hoặc `/tmp/`.

### 3. Chạy Benchmark GUI
Khởi động giao diện web đánh giá và đối soát mô hình:

```bash
python benchmark/benchmark.py
```
- Trình duyệt sẽ tự động mở tại `http://127.0.0.1:8765`.
- Hỗ trợ nhập review mới để dự đoán tức thì hoặc duyệt tập test với bảng số liệu F1-score/MAE theo từng khía cạnh.

### 4. Chạy Ứng dụng Desktop
Khởi động ứng dụng giao diện native (macOS / Windows):

```bash
# Khởi chạy chế độ phát triển
npm --prefix desktop run desktop

# Đóng gói installer (tuỳ chọn)
npm --prefix desktop run dist:mac  # Build DMG/ZIP trên macOS
npm --prefix desktop run dist:win  # Build EXE/NSIS trên Windows
```

---

## 📝 Viết LaTeX & Quản lý Trích dẫn

Toàn bộ mã nguồn bài báo học thuật theo chuẩn **Springer Nature** nằm tại thư mục [docs/main_paper/](docs/main_paper/).

### 1. Biên dịch bài báo
Hệ thống đã tích hợp sẵn script tự động hóa [build.sh](docs/main_paper/build.sh) (chạy 3-pass `pdflatex` + `bibtex` và tự dọn dẹp các file rác):

```bash
# Biên dịch ra file PDF hoàn chỉnh:
bash docs/main_paper/build.sh

# Hoặc chỉ định rõ file nguồn nếu có thay đổi:
bash docs/main_paper/build.sh sn-article.tex

# Dọn dẹp nhanh các file phụ trợ (.aux, .log, .bbl, .out, ...):
bash docs/main_paper/build.sh --clean
```
File PDF đầu ra sẽ nằm tại `docs/main_paper/sn-article.pdf`.

### 2. Quản lý Tài liệu tham khảo & Trích dẫn (Citations)

- **Kho bài báo gốc**: Tất cả 32 bài báo tham khảo đã tải về dạng PDF và đánh số thứ tự tại `docs/src_paper/[1-32].pdf`.
- **File BibTeX**: Quản lý thông tin trích dẫn tập trung tại [docs/main_paper/sn-bibliography.bib](docs/main_paper/sn-bibliography.bib).

#### Quy trình trích dẫn tài liệu:
1. **Thêm trích dẫn mới vào `.bib`**:
   Mở `docs/main_paper/sn-bibliography.bib` và thêm mục mới:
   ```bibtex
   %% Reference [33] (tương ứng file docs/src_paper/33.pdf)
   @article{tacgia2026tieude,
     author  = {Nguyen, Van A and Le, Thi B},
     title   = {Recent Advances in Aspect-Based Sentiment Analysis},
     journal = {IEEE Transactions on Affective Computing},
     year    = {2026},
     doi     = {10.xxxx/xxxx}
   }
   ```
2. **Trích dẫn trong văn bản LaTeX**:
   Dùng lệnh `\cite{citation_key}` tại vị trí cần dẫn chứng trong `sn-article.tex`:
   ```latex
   Phương pháp này kế thừa kiến trúc từ nghiên cứu trước \cite{tacgia2026tieude}.
   Hoặc trích dẫn nhiều bài cùng lúc: \cite{bimaputra2023aspect, bruni2025amaqa}.
   ```
3. **Đồng bộ danh mục tài liệu**:
   Nếu file `sn-article.tex` sử dụng khối `\begin{thebibliography}{99}`, hãy bổ sung tương ứng:
   ```latex
   \bibitem[33]{tacgia2026tieude}
   Nguyen, V.~A., \& Le, T.~B.
   Recent Advances in Aspect-Based Sentiment Analysis.
   \textit{IEEE Transactions on Affective Computing} (2026).
   ```
   Sau đó chạy lại `bash docs/main_paper/build.sh` để cập nhật số thứ tự tham chiếu.
