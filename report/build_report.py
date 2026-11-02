"""Build the final report PDF: figures -> HTML (with table-of-contents page numbers) -> PDF via headless Chrome.

Usage:  python build_report.py
Needs Google Chrome or Microsoft Edge, matplotlib and PyMuPDF (fitz).
"""
import os, re, subprocess, sys, shutil
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "report_template.html")
HTML = os.path.join(HERE, "report.html")
PDF = os.path.join(HERE, "Final_Report_ASR_to_MT.pdf")

SECTIONS = {
    "s1": "1. Problem statement", "s2": "2. Datasets used", "s3": "3. Models used and system built",
    "s4": "4. Methodology implemented", "s5": "5. Results", "s6": "6. Analysis",
    "s7": "7. Comparison with the reference papers", "s8": "8. Limitations", "s9": "9. Future scope",
    "s10": "10. Conclusion", "refs": "References", "appA": "Appendix A. Implementation log and issues resolved",
    "appB": "Appendix B. Worked examples of ASR-induced translation errors",
    "appC": "Appendix C. Metric definitions and file inventory",
}
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def print_pdf(html_path, pdf_path):
    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        sys.exit("Chrome or Edge not found")
    url = "file:///" + html_path.replace("\\", "/").replace(" ", "%20")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    "--virtual-time-budget=8000", f"--print-to-pdf={pdf_path}", url], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)


def build(page_numbers):
    html = open(TEMPLATE, encoding="utf-8").read()
    html = re.sub(r"\{\{pg:(\w+)\}\}", lambda m: str(page_numbers.get(m.group(1), "")), html)
    open(HTML, "w", encoding="utf-8").write(html)
    print_pdf(HTML, PDF)


def find_pages():
    doc = fitz.open(PDF)
    found = {}
    for key, title in SECTIONS.items():
        for i in range(3, len(doc)):        # pages 0-2: cover, abstract, contents
            hit = False
            for block in doc[i].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    big = max(s["size"] for s in line["spans"]) >= 15      # chapter headings are 17 pt
                    if big and title.startswith(text[:25]) and text[:25] == title[:25]:
                        hit = True
            if hit:
                found[key] = i + 1
                break
    return found, len(doc)


if __name__ == "__main__":
    subprocess.run([sys.executable, os.path.join(HERE, "make_figures.py")], check=True)
    build({})
    pages, n = find_pages()
    build(pages)
    pages2, n2 = find_pages()
    print("pages:", n2, "| toc numbers stable:", pages == pages2)
    print(pages2)
    print("wrote", PDF)
