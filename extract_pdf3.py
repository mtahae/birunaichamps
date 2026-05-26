import fitz
import os

pdf_path = r"c:\Users\tahaee\Desktop\birun\Coklu_Veri_Seti_Stratejisi.pdf"
out_path = r"c:\Users\tahaee\Desktop\birun\outputs\coklu_veri_seti.txt"

doc = fitz.open(pdf_path)
text = "\n".join(p.get_text() for p in doc)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"Extracted {pdf_path} ({len(text)} chars)")
