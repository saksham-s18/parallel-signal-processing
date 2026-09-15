import os
import re
import base64
import json
import time
import subprocess
import threading
import http.server
import socketserver
import markdown
import pypdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
MD_PATH = os.path.join(REPORT_DIR, 'report.md')
HTML_PATH = os.path.join(REPORT_DIR, 'report.html')
PDF_PATH = os.path.join(REPORT_DIR, 'Final_Report.pdf')

def start_local_server(directory):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), QuietHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port

def read_and_convert_md():
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Pre-process image paths to base64 data URIs so they are 100% self-contained
    def img_replace(match):
        alt = match.group(1)
        rel_path = match.group(2)
        img_full_path = os.path.abspath(os.path.join(REPORT_DIR, rel_path))
        if os.path.exists(img_full_path):
            with open(img_full_path, 'rb') as img_f:
                b64 = base64.b64encode(img_f.read()).decode('utf-8')
            ext = os.path.splitext(img_full_path)[1].lstrip('.').lower()
            if ext == 'jpg': ext = 'jpeg'
            return f'<div class="figure-container"><img alt="{alt}" src="data:image/{ext};base64,{b64}" /><p class="figure-caption"><strong>{alt}</strong></p></div>'
        else:
            print(f"Warning: Image not found: {img_full_path}")
            return match.group(0)

    # Insert page break after Front Matter so the title & metadata are prominently placed on Page 1
    pattern = r'(\|\s*\*\*Demonstration Link\*\*.*?\|\s*\n\s*---\s*\n)'
    md_text = re.sub(pattern, r'\1\n<div class="page-break"></div>\n\n', md_text, flags=re.DOTALL)

    # Insert page break before Section 18 so Contributions section sits cleanly on its own final page
    md_text = re.sub(r'(\n---\s*\n\s*## 18\. Contributions)', r'\n<div class="page-break"></div>\n\n## 18. Contributions', md_text)

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_replace, md_text)

    # Convert markdown to html with tables and fenced code
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'nl2br'])

    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Parallel Signal Processing: Moving-Average Noise Reduction</title>

<!-- MathJax with CommonHTML for crisp mathematical rendering -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  options: {{
    renderActions: {{
      addMenu: []
    }}
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

<style>
@page {{
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 9pt;
  line-height: 1.5;
  color: #1a202c;
  background: #ffffff;
  margin: 0;
  padding: 0;
}}

/* Typography */
h1 {{
  font-size: 20pt;
  font-weight: 800;
  color: #0f172a;
  text-align: center;
  margin-top: 25px;
  margin-bottom: 25px;
  line-height: 1.25;
  letter-spacing: -0.3px;
}}

h2 {{
  font-size: 12.5pt;
  font-weight: 700;
  color: #1e3a8a;
  border-bottom: 1.5px solid #cbd5e1;
  padding-bottom: 4px;
  margin-top: 22px;
  margin-bottom: 10px;
  page-break-after: avoid;
}}

h3 {{
  font-size: 10.5pt;
  font-weight: 700;
  color: #334155;
  margin-top: 14px;
  margin-bottom: 6px;
  page-break-after: avoid;
}}

h4 {{
  font-size: 9.5pt;
  font-weight: 700;
  color: #475569;
  margin-top: 10px;
  margin-bottom: 4px;
  page-break-after: avoid;
}}

p {{
  margin-top: 0;
  margin-bottom: 8px;
  text-align: justify;
}}

ul, ol {{
  margin-top: 0;
  margin-bottom: 8px;
  padding-left: 20px;
}}

li {{
  margin-bottom: 3px;
}}

/* Tables */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 8pt;
  page-break-inside: avoid;
}}

th, td {{
  border: 1px solid #cbd5e1;
  padding: 5px 7px;
  text-align: left;
  vertical-align: middle;
}}

th {{
  background-color: #1e293b;
  color: #ffffff;
  font-weight: 600;
  letter-spacing: 0.1px;
}}

tr:nth-child(even) {{
  background-color: #f8fafc;
}}

/* Preformatted and Code Blocks */
pre {{
  background-color: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3.5px solid #2563eb;
  border-radius: 4px;
  padding: 8px 10px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 7.5pt;
  line-height: 1.4;
  overflow-x: auto;
  page-break-inside: avoid;
  margin: 10px 0;
}}

code {{
  font-family: Consolas, "Courier New", monospace;
  font-size: 8.2pt;
  background-color: #f1f5f9;
  padding: 1px 3px;
  border-radius: 3px;
  color: #0f172a;
}}

pre code {{
  background-color: transparent;
  padding: 0;
  color: inherit;
}}

/* Figures and Images */
.figure-container {{
  page-break-inside: avoid;
  margin: 14px auto;
  text-align: center;
}}

img {{
  max-width: 86%;
  height: auto;
  display: block;
  margin: 0 auto 4px auto;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

.figure-caption {{
  font-size: 8pt;
  color: #475569;
  margin-top: 4px;
  text-align: center;
}}

/* Alerts and Blockquotes */
blockquote {{
  border-left: 3.5px solid #2563eb;
  background-color: #eff6ff;
  color: #1e40af;
  margin: 10px 0;
  padding: 6px 12px;
  border-radius: 0 4px 4px 0;
  font-size: 8.5pt;
}}

hr {{
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 18px 0;
}}

.page-break {{
  page-break-after: always;
  height: 0;
  margin: 0;
  padding: 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Generated HTML at: {HTML_PATH}")
    return HTML_PATH

def generate_pdf(server_port):
    edge_paths = [
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    ]
    browser_exe = next((p for p in edge_paths if os.path.exists(p)), None)
    if not browser_exe:
        raise RuntimeError("Neither Microsoft Edge nor Google Chrome found.")

    url = f'http://127.0.0.1:{server_port}/report/report.html'
    cmd = [
        browser_exe,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--virtual-time-budget=6000',
        '--run-all-compositor-stages-before-draw',
        '--print-to-pdf=' + PDF_PATH,
        url
    ]

    print(f"Executing: {browser_exe} to generate PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Browser stderr: {res.stderr}")
        raise RuntimeError(f"Browser failed with return code {res.returncode}")
    print(f"PDF successfully written to: {PDF_PATH}")

def verify_pdf(pdf_file):
    print("\n=======================================================")
    print(" VERIFYING FINAL GENERATED PDF REPORT")
    print("=======================================================")
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF does not exist: {pdf_file}")
    
    file_size_kb = os.path.getsize(pdf_file) / 1024.0
    file_size_mb = file_size_kb / 1024.0
    print(f"PDF File Path : {pdf_file}")
    print(f"PDF File Size : {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")

    reader = pypdf.PdfReader(pdf_file)
    total_pages = len(reader.pages)
    print(f"Total Pages   : {total_pages}")

    # Check Page 1 Metadata
    page1_text = reader.pages[0].extract_text()
    required_metadata = [
        ("Application Theme", "Signal Processing"),
        ("Assigned Problem", "Moving-Average Noise Reduction"),
        ("Group Number", "16"),
        ("Group Leader", "Saksham Singh"),
        ("Roll No (Leader)", "2024BCS0070"),
        ("Member 2", "Daksh Singh"),
        ("Roll No (Member 2)", "2024BCS0042"),
        ("Member 3", "Anmol Pipara"),
        ("Roll No (Member 3)", "2024BCS0014"),
        ("Google Drive Link", "1i3eLAEA6jXPNDnx8pnbJffigIk-LwEPE")
    ]
    
    print("\n[CHECK 1] Page 1 Assignment Metadata:")
    all_meta_ok = True
    for label, val in required_metadata:
        found = val in page1_text
        print(f"  - {label} ({val}): {'[OK] PRESENT' if found else '[FAIL] MISSING'}")
        if not found: all_meta_ok = False

    # Check all 18 Sections
    full_doc_text = ""
    for idx, page in enumerate(reader.pages):
        full_doc_text += page.extract_text() + "\n"

    sections_to_check = [
        "1. Problem Statement", "2. Objective", "3. Application and Parallelization",
        "4. Algorithm", "5. Input Data and Test Cases", "6. Correctness Verification",
        "7. Experimental Environment", "8. Benchmark Methodology", "9. Performance Results",
        "10. Performance Analysis", "11. CUDA Optimization Analysis",
        "12. Sequential vs OpenMP vs CUDA vs Optimized CUDA",
        "13. When Parallelization Does Not Help", "14. Reproducibility",
        "15. Figures", "16. Limitations and Observations", "17. Conclusion",
        "18. Contributions"
    ]

    print("\n[CHECK 2] Report Structure (All 18 Sections):")
    all_sec_ok = True
    for sec in sections_to_check:
        found = sec in full_doc_text
        print(f"  - Section {sec}: {'[OK]' if found else '[FAIL]'}")
        if not found: all_sec_ok = False

    # Check Benchmark Tables & Implementations
    print("\n[CHECK 3] Key Benchmark Tables & Numbers:")
    key_metrics = [
        ("Sequential Baseline", r"Sequential"),
        ("OpenMP 8 Threads", r"OpenMP\s*\([^\)]*8"),
        ("OpenMP 24 Threads", r"OpenMP\s*\([^\)]*24"),
        ("CUDA Baseline", r"CUDA Baseline"),
        ("CUDA Optimized", r"CUDA Optimized"),
        ("Small Seq Time (0.1290 ms)", r"0\.1290"),
        ("Medium Seq Time (1.6540 ms)", r"1\.6540"),
        ("Large Seq Time (39.5710 ms)", r"39\.5710"),
        ("Large Kernel Speedup (25.05x)", r"25\.05x"),
        ("Large End-to-End Speedup (13.89x)", r"13\.89x"),
        ("Host-to-Device Transfer (H2D)", r"H2D"),
        ("Device-to-Host Transfer (D2H)", r"D2H")
    ]
    all_metrics_ok = True
    for label, pattern in key_metrics:
        found = bool(re.search(pattern, full_doc_text))
        print(f"  - {label}: {'[OK]' if found else '[FAIL]'}")
        if not found: all_metrics_ok = False

    # Check Embedded Images
    print("\n[CHECK 4] Embedded Figures & Plots:")
    total_images = sum(len(page.images) for page in reader.pages)
    print(f"  - Total embedded figure images in PDF: {total_images} (Required: 5)")
    images_ok = (total_images >= 5)

    # Check Section 18 Clean Layout (No legacy tables / columns)
    print("\n[CHECK 5] Exclusion of legacy form fields in Section 18:")
    banned_in_s18 = ["Contribution (%)", "Signature", "Semester & Batch", "Submission Date"]
    s18_text = full_doc_text[full_doc_text.find("18. Contributions"):] if "18. Contributions" in full_doc_text else ""
    exclusions_ok = True
    for banned in banned_in_s18:
        present = banned in s18_text
        print(f"  - '{banned}' absent: {'[OK] ABSENT' if not present else '[FAIL] FOUND'}")
        if present: exclusions_ok = False

    print("\n=======================================================")
    overall = all_meta_ok and all_sec_ok and all_metrics_ok and images_ok and exclusions_ok
    print(f" OVERALL PDF VERIFICATION STATUS: {'SUCCESS [PASSED]' if overall else 'FAILED'}")
    print("=======================================================\n")

if __name__ == '__main__':
    read_and_convert_md()
    server, server_port = start_local_server(PROJECT_ROOT)
    print(f"Local HTTP server running on http://127.0.0.1:{server_port}")
    try:
        generate_pdf(server_port)
        verify_pdf(PDF_PATH)
    finally:
        server.shutdown()
