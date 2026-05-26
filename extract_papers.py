import fitz
import os

papers_dir = r"c:\Users\tahaee\Desktop\birun\datasets\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\papers"
out_dir = r"c:\Users\tahaee\Desktop\birun\outputs\papers_text"
os.makedirs(out_dir, exist_ok=True)

# Ana makale + ilk sıralarda yer alan büyük/önemli makaleler
priority = [
    "2020ChallengePaper.pdf",
    "CinC2020-305.pdf",   # en büyük (2MB)
    "CinC2020-138.pdf",   # ikinci büyük (1.7MB)
    "CinC2020-353.pdf",   # üçüncü büyük (1.6MB)
    "CinC2020-374.pdf",   # (1.6MB)
    "CinC2020-198.pdf",   # (1.2MB)
    "CinC2020-225.pdf",   # (1.1MB)
    "CinC2020-297.pdf",   # (940KB)
    "CinC2020-107.pdf",   # (951KB)
    "CinC2020-281.pdf",   # (690KB)
]

for fname in priority:
    path = os.path.join(papers_dir, fname)
    out_path = os.path.join(out_dir, fname.replace(".pdf", ".txt"))
    try:
        doc = fitz.open(path)
        text = "\n".join(p.get_text() for p in doc)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"OK: {fname} ({len(text)} chars)")
    except Exception as e:
        print(f"ERR: {fname}: {e}")
