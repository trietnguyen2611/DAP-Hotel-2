# CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment

Research and application system for multi-aspect hotel review evaluation (**Aspect-Based Sentiment Analysis - ABSA**), rationale sentence extraction, and cross-aspect leakage control based on the **CRAM-ABSA** architecture (*Contrastive Rationale-Aligned Multi-Task Architecture with Embedding-Based Retrieval - EBR*).

- **GitHub Repository:** [https://github.com/trietnguyen2611/CRAM-ABSA-EBR-Hotel-Reviews](https://github.com/trietnguyen2611/CRAM-ABSA-EBR-Hotel-Reviews)
- **Research Paper:** Springer Nature template (`docs/main_paper/sn-article.tex` -> `CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment.pdf`)

---

## 🌟 Key Features

1. **6 Standardized Aspects:**
   - `Service`: Customer service, staff attitude, front desk, housekeeping.
   - `Facility`: Room amenities, bathroom, cleanliness, soundproofing, air conditioning.
   - `Experience`: Overall guest stay experience.
   - `Amenity`: Hotel amenities (swimming pool, buffet/dining, gym, spa, Wi-Fi, parking).
   - `Loyalty`: Customer loyalty, likelihood to return or recommend to others.
   - `Branding`: Pricing, brand reputation, value for money.

2. **Advanced Dual-Stage Pipeline:**
   - **Stage 1 (Linguistic & Dense Retrieval):** Combines grammatical analysis, negation handling (NLTK), and semantic retrieval using Embedding-Based Retrieval (EBR) to extract rationale sentences for each aspect.
   - **Stage 2 (Multi-Task BERT Backbone):** Integrates 20 tabular context features through Residual Gating, uses an Orthogonal Gram Penalty to prevent cross-aspect semantic leakage (leakage drops to **0.24%**), and uses Conditional Ordinal Regression (CORN) to maintain the natural order of star ratings.

3. **High Performance on Test Set (999 reviews from 9,990 TripAdvisor reviews):**
   - **Test Loss:** `0.2552` (reduced by over 50% compared to baselines `0.5143 - 0.6032`).
   - **6-Aspect Macro-F1:** `72.55%` (+20.00% over BERT-Base, +20.43% over WSD-BiLSTM-GCN).
   - **Macro-MAE:** `0.3520`.
   - **Rationale F1:** `70.84%` (+20.18% to +61.07% over baseline models).
   - **Acc@0:** `92.39%`.

---

## 📁 Project Structure

```text
├── data/
│   ├── dataset/         # Original TripAdvisor dataset (TripAdvisor_EN.json - 44MB)
│   └── df/              # Processed data: hotel_reviews.parquet (42 columns, 9,990 samples) & chatbot.db (SQLite)
├── train/
│   ├── train_model/     # Base model training notebook (trainv1.ipynb)
│   └── train_method/    # Ablation studies and method variants training (trainmethod.ipynb)
├── model/
│   ├── modelv1/         # Checkpoints: modelv1.pt, ebr_exemplar_bank.pkl & training_history_v1.json
│   └── method/          # Method comparison checkpoints (modelmethod1.pt, modelmethod9.pt,...)
├── benchmark/
│   └── benchmark.py     # Web GUI for performance comparison, single-review testing, and visualization
├── desktop/             # Commercial Desktop app (Electron + Python Backend + Vanilla CSS/JS)
├── docs/
│   ├── main_paper/      # Springer Nature LaTeX paper source (sn-article.tex, build.sh, .bib, chart_img/)
│   ├── src_paper/       # 30 foundational research reference papers (1.pdf -> 30.pdf)
│   └── Springer_Nature_LaTeX_Template/ # Official Springer Nature template files
├── Project.ipynb        # Data preprocessing pipeline, feature extraction, and EDA charts
└── requirements.txt     # Required Python packages
```

---

## ⚙️ Environment Setup

### 1. Python Virtual Environment Setup
Python **3.10 – 3.12** is required:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment:
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Desktop App Environment Setup
Node.js **20+** is required (if you want to run or package the Desktop app):

```bash
npm --prefix desktop install
```

---

## 🚀 Execution Guide

### 1. Data Processing & Exploratory Data Analysis (EDA)
Open [Project.ipynb](Project.ipynb) in VSCode or Jupyter Lab to run the preprocessing pipeline:
- Load and inspect the dataset `data/dataset/TripAdvisor_EN.json`.
- Dual-stream text cleaning: combines linguistic lemmatization with BERT token structure preservation.
- Extract 20 contextual features and 6-aspect rating labels.
- Export cleaned data to `data/df/hotel_reviews.parquet` (9,990 rows) and save to database `data/df/chatbot.db`.
- Automatically save all generated charts to `docs/main_paper/chart_img/`.

### 2. Model Training
The Jupyter Notebooks in the `train/` directory can run locally (GPU / Apple Silicon MPS / CPU) or on Kaggle / Google Colab:
- **[train/train_model/trainv1.ipynb](train/train_model/trainv1.ipynb)**: Train the core CRAM-ABSA architecture with BERT Backbone across 6 standardized aspects (`Service`, `Facility`, `Experience`, `Amenity`, `Loyalty`, `Branding`), Rationale Extraction, and EBR.
- **[train/train_method/trainmethod.ipynb](train/train_method/trainmethod.ipynb)**: Train baseline comparison models and ablation variants (Contrastive Loss, loss weighting schemes).

### 3. Run Benchmark Web GUI
The project includes an interactive Web Dashboard to compare models and test reviews:

```bash
python benchmark/benchmark.py
```
- Open `http://127.0.0.1:8765` in your browser.
- Enter any custom review text to get real-time sentiment predictions across 6 aspects and extracted rationale evidence.
- Evaluate the entire test set (999 reviews), view confusion matrices, error distribution charts, and key metrics (Macro-F1, Macro-MAE, Acc@0, Rationale F1).

### 4. Run Desktop Application
A complete cross-platform desktop app with a modern UI connected to the AI backend:

```bash
# Run the application in development mode:
npm --prefix desktop run desktop

# Package the application:
npm --prefix desktop run dist:mac  # Build DMG / ZIP (macOS)
npm --prefix desktop run dist:win  # Build EXE / NSIS installer (Windows)
```

---

## 📝 Compiling Research Paper (Springer Nature)

All academic paper files follow the official **Springer Nature** template in `docs/main_paper/`.

### 1. Compile to PDF
An automated compilation script `build.sh` is included (runs 3-pass `pdflatex` + `bibtex`):

```bash
# Build the complete PDF document:
bash docs/main_paper/build.sh

# Or compile a specific source file:
bash docs/main_paper/build.sh sn-article.tex

# Clean up log and temporary files (.aux, .bbl, .blg, .out, .toc,...):
bash docs/main_paper/build.sh --clean
```
Output PDF file: `docs/main_paper/CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment.pdf`.

### 2. Reference Management
- All **30 reference papers** are stored in `docs/src_paper/` (`1.pdf` to `30.pdf`).
- Citation details are in `docs/main_paper/sn-bibliography.bib` with indexed keys (`ref01_...` to `ref30_...`), matching the citation order in the paper.
