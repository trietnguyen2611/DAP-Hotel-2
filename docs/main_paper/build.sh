#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PAPER_DIR}"

TEX_FILE="${1:-sn-article.tex}"
JOB_NAME="${TEX_FILE%.tex}"

cleanup_aux() {
    echo "🧹 Đang dọn dẹp các file tạm / thừa sau khi compile..."
    rm -f \
        "${JOB_NAME}.aux" \
        "${JOB_NAME}.bbl" \
        "${JOB_NAME}.blg" \
        "${JOB_NAME}.bcf" \
        "${JOB_NAME}.run.xml" \
        "${JOB_NAME}.log" \
        "${JOB_NAME}.out" \
        "${JOB_NAME}.toc" \
        "${JOB_NAME}.lof" \
        "${JOB_NAME}.lot" \
        "${JOB_NAME}.fls" \
        "${JOB_NAME}.fdb_latexmk" \
        "${JOB_NAME}.synctex.gz" \
        "${JOB_NAME}.synctex(busy)" \
        "${JOB_NAME}.nav" \
        "${JOB_NAME}.snm" \
        "${JOB_NAME}.vrb" \
        "${JOB_NAME}.xdv" \
        "${JOB_NAME}.spl" \
        "${JOB_NAME}.brf" \
        "${JOB_NAME}.idx" \
        "${JOB_NAME}.ilg" \
        "${JOB_NAME}.ind" \
        "${JOB_NAME}.acn" \
        "${JOB_NAME}.acr" \
        "${JOB_NAME}.alg" \
        "${JOB_NAME}.glg" \
        "${JOB_NAME}.glo" \
        "${JOB_NAME}.gls" \
        "${JOB_NAME}.ist" \
        texput.log
}

if [[ "${1:-}" == "--clean" ]]; then
    JOB_NAME="sn-article"
    cleanup_aux
    rm -f "${PAPER_DIR}"/*.aux
    echo "✅ Đã xoá toàn bộ các file phụ trợ trong: ${PAPER_DIR}"
    exit 0
fi

if [[ ! -f "${TEX_FILE}" ]]; then
    echo "❌ Không tìm thấy file nguồn LaTeX: ${PAPER_DIR}/${TEX_FILE}"
    exit 1
fi

echo "📄 Đang biên dịch LaTeX: ${PAPER_DIR}/${TEX_FILE} ..."

export PATH="/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "❌ Không tìm thấy trình biên dịch pdflatex trong PATH."
    exit 1
fi

echo "  ➜ Pass 1/3: pdflatex..."
pdflatex -interaction=nonstopmode -halt-on-error "${TEX_FILE}" >/dev/null

if grep -q "\\\\bibdata" "${JOB_NAME}.aux" 2>/dev/null && ! grep -q "\\\\begin{thebibliography}" "${TEX_FILE}"; then
    if command -v bibtex >/dev/null 2>&1; then
        echo "  ➜ Chạy bibtex..."
        bibtex "${JOB_NAME}" >/dev/null || true
    fi
fi

echo "  ➜ Pass 2/3: pdflatex (cập nhật tham chiếu & mục lục)..."
pdflatex -interaction=nonstopmode -halt-on-error "${TEX_FILE}" >/dev/null

echo "  ➜ Pass 3/3: pdflatex (hoàn thiện layout)..."
pdflatex -interaction=nonstopmode -halt-on-error "${TEX_FILE}" >/dev/null

cleanup_aux

if [[ -f "${JOB_NAME}.pdf" ]]; then
    OUTPUT_PDF="${JOB_NAME}.pdf"
    if [[ "${TEX_FILE}" == "sn-article.tex" ]]; then
        OUTPUT_PDF="CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment.pdf"
        mv -f "${JOB_NAME}.pdf" "${OUTPUT_PDF}"
    fi
    PDF_SIZE="$(du -h "${OUTPUT_PDF}" | cut -f1 | xargs)"
    echo "✅ Biên dịch hoàn tất: ${PAPER_DIR}/${OUTPUT_PDF} (${PDF_SIZE})"
else
    echo "❌ Biên dịch thất bại: Không tạo được ${JOB_NAME}.pdf"
    exit 1
fi
