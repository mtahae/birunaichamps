import fitz
import os

pdfs = [
    r"c:\Users\tahaee\Desktop\birun\Between_a_ROC_and_a_heart_place\Between_a_ROC_and_a_heart_place\112_CinCFinalPDF.pdf",
    r"c:\Users\tahaee\Desktop\birun\Triage\Triage\133_CinCFinalPDF.pdf",
    r"c:\Users\tahaee\Desktop\birun\SharifAITeam\SharifAITeam\445_CinCFinalPDF.pdf"
]

out_dir = r"c:\Users\tahaee\Desktop\birun\outputs\papers_text"
os.makedirs(out_dir, exist_ok=True)

for pdf_path in pdfs:
    fname = os.path.basename(pdf_path)
    out_path = os.path.join(out_dir, fname.replace(".pdf", ".txt"))
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(p.get_text() for p in doc)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {fname}")
    except Exception as e:
        print(f"Failed to extract {fname}: {e}")
