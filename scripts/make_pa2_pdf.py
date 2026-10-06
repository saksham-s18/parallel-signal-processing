import os
import re
import base64
import json
import time
import subprocess
import threading
import http.server
import socketserver
import shutil
import markdown
import pypdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
MD_PATH = os.path.join(REPORT_DIR, 'report_pa2.md')
HTML_PATH = os.path.join(REPORT_DIR, 'report_pa2.html')
PDF_PATH_PA2 = os.path.join(REPORT_DIR, 'report_pa2.pdf')
PDF_PATH_FINAL = os.path.join(REPORT_DIR, 'PA2_Final_Report.pdf')

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

    # In-place base64 replacement for the cover logo
    logo_path = os.path.join(REPORT_DIR, 'iiitk_logo.png')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f_logo:
            logo_b64 = base64.b64encode(f_logo.read()).decode('utf-8')
        md_text = md_text.replace('src="iiitk_logo.png"', f'src="data:image/png;base64,{logo_b64}"')

    # Insert page break before Section 17 so Contributions section sits cleanly on its own final page
    md_text = re.sub(r'(\n---\s*\n\s*## 17\. Contributions)', r'\n<div class="page-break"></div>\n\n## 17. Contributions', md_text)

    # Also clean section dividers before major figures for clean pagination
    md_text = re.sub(r'(\n## 7\. MPI Strong Scaling)', r'\n<div class="page-break"></div>\n\n## 7. MPI Strong Scaling', md_text)
    md_text = re.sub(r'(\n## 10\. Hardware Energy)', r'\n<div class="page-break"></div>\n\n## 10. Hardware Energy', md_text)
    md_text = re.sub(r'(\n## 11\. Comprehensive Six-Way)', r'\n<div class="page-break"></div>\n\n## 11. Comprehensive Six-Way', md_text)

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_replace, md_text)

    # Convert markdown to html with tables and fenced code
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'nl2br'])

    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PA2 Final Report: Parallel Signal Processing Moving-Average Filter</title>

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
  margin: 16mm 14mm 16mm 14mm;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 8.7pt;
  line-height: 1.45;
  color: #1a202c;
  background: #ffffff;
  margin: 0;
  padding: 0;
}}

/* Cover Page (Matching University Template) */
.cover-page {{
  padding-top: 35px;
  text-align: center;
}}

.cover-course {{
  font-size: 17pt;
  font-weight: 800;
  color: #000000;
  letter-spacing: 0.2px;
  margin-bottom: 20px;
  text-align: center;
}}

.cover-assignment {{
  font-size: 13pt;
  font-weight: 700;
  color: #000000;
  letter-spacing: 0.3px;
  margin-bottom: 5px;
  text-align: center;
}}

.cover-report-type {{
  font-size: 11pt;
  font-weight: 500;
  color: #222222;
  margin-bottom: 35px;
  text-align: center;
}}

.cover-logo-wrapper {{
  margin: 0 auto 35px auto;
  text-align: center;
}}

.cover-logo {{
  width: 206px;
  max-width: 210px;
  height: auto;
  display: block;
  margin: 0 auto;
}}

.cover-table {{
  width: 96%;
  max-width: 660px;
  margin: 0 auto;
  border-collapse: collapse;
  font-size: 8.8pt;
  border: 1px solid #111827;
}}

.cover-table tr {{
  background-color: #ffffff !important;
}}

.cover-table td {{
  border: 1px solid #111827 !important;
  padding: 8px 12px;
  vertical-align: middle;
  text-align: left;
}}

.cover-field-label {{
  font-weight: 700;
  color: #000000;
  width: 36%;
}}

.cover-field-value {{
  color: #111827;
  width: 64%;
  word-break: break-all;
}}

/* Typography */
h1 {{
  font-size: 19pt;
  font-weight: 800;
  color: #0f172a;
  text-align: center;
  margin-top: 20px;
  margin-bottom: 8px;
  line-height: 1.25;
  letter-spacing: -0.3px;
}}

h1 + h2 {{
  font-size: 12.5pt;
  font-weight: 600;
  color: #2563eb;
  text-align: center;
  margin-top: 0;
  margin-bottom: 4px;
  border-bottom: none;
}}

h1 + h2 + h3 {{
  font-size: 9.5pt;
  font-weight: 400;
  color: #64748b;
  text-align: center;
  margin-top: 0;
  margin-bottom: 20px;
}}

h2 {{
  font-size: 12pt;
  font-weight: 700;
  color: #1e3a8a;
  border-bottom: 1.5px solid #cbd5e1;
  padding-bottom: 3px;
  margin-top: 18px;
  margin-bottom: 8px;
  page-break-after: avoid;
}}

h3 {{
  font-size: 10pt;
  font-weight: 700;
  color: #334155;
  margin-top: 12px;
  margin-bottom: 5px;
  page-break-after: avoid;
}}

h4 {{
  font-size: 9pt;
  font-weight: 700;
  color: #475569;
  margin-top: 8px;
  margin-bottom: 3px;
  page-break-after: avoid;
}}

p {{
  margin-top: 0;
  margin-bottom: 6px;
  text-align: justify;
}}

ul, ol {{
  margin-top: 0;
  margin-bottom: 6px;
  padding-left: 18px;
}}

li {{
  margin-bottom: 2px;
}}

/* Tables */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
  font-size: 7.6pt;
  page-break-inside: avoid;
}}

th, td {{
  border: 1px solid #cbd5e1;
  padding: 4px 6px;
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
  padding: 6px 8px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 7.2pt;
  line-height: 1.35;
  overflow-x: auto;
  page-break-inside: avoid;
  margin: 8px 0;
}}

code {{
  font-family: Consolas, "Courier New", monospace;
  font-size: 7.8pt;
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

.listing-caption {{
  font-size: 8.2pt;
  font-weight: 700;
  color: #1e3a8a;
  margin-top: 8px;
  margin-bottom: 2px;
  text-align: left;
  page-break-after: avoid;
}}

/* Figures and Images */
.figure-container {{
  page-break-inside: avoid;
  margin: 10px auto;
  text-align: center;
}}

img {{
  max-width: 84%;
  height: auto;
  display: block;
  margin: 0 auto 3px auto;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

.figure-caption {{
  font-size: 7.5pt;
  color: #475569;
  margin-top: 3px;
  text-align: center;
}}

/* Alerts and Blockquotes */
blockquote {{
  border-left: 3.5px solid #2563eb;
  background-color: #eff6ff;
  color: #1e40af;
  margin: 8px 0;
  padding: 5px 10px;
  border-radius: 0 4px 4px 0;
  font-size: 8.2pt;
}}

hr {{
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 14px 0;
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

    url = f'http://127.0.0.1:{server_port}/report/report_pa2.html'
    cmd = [
        browser_exe,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--no-pdf-header-footer',
        '--virtual-time-budget=6000',
        '--run-all-compositor-stages-before-draw',
        '--print-to-pdf=' + PDF_PATH_PA2,
        url
    ]

    print(f"Executing browser to generate PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Browser stderr: {res.stderr}")
        raise RuntimeError(f"Browser failed with return code {res.returncode}")
    print(f"PA2 PDF successfully written to: {PDF_PATH_PA2}")

    # Copy to PA2_Final_Report.pdf as well
    try:
        shutil.copy2(PDF_PATH_PA2, PDF_PATH_FINAL)
    except PermissionError:
        print("Acrobat is locking PA2_Final_Report.pdf, terminating Acrobat to release lock...")
        subprocess.run(["powershell", "-Command", "Stop-Process -Name Acrobat -Force -ErrorAction SilentlyContinue"])
        time.sleep(1)
        shutil.copy2(PDF_PATH_PA2, PDF_PATH_FINAL)
    print(f"PA2 PDF also saved as: {PDF_PATH_FINAL}")

def verify_pdf(pdf_file):
    print("\n=======================================================")
    print(" VERIFYING FINAL GENERATED PA2 PDF REPORT")
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
        ("Course Title", "CSS311 – PARALLEL & DISTRIBUTED COMPUTING"),
        ("Assignment Name", "PROGRAMMING ASSIGNMENT – 2"),
        ("Report Title", "PA2 Comprehensive Final Report"),
        ("Group Number", "16"),
        ("Group Leader", "2024BCS0070 – Saksham Singh"),
        ("Member 2", "2024BCS0042 – Daksh Singh"),
        ("Member 3", "2024BCS0014 – Anmol Pipara"),
        ("Semester & Batch", "Semester 5, Batch 3"),
        ("Application Theme", "Signal Processing"),
        ("Assigned Problem", "1D Moving-Average Noise Reduction Filter Across Six Parallel Paradigms")
    ]
    
    print("\n[CHECK 1] Page 1 Assignment Metadata (Matching University Template):")
    all_meta_ok = True
    for label, val in required_metadata:
        found = val in page1_text
        print(f"  - {label} ({val}): {'[OK] PRESENT' if found else '[FAIL] MISSING'}")
        if not found: all_meta_ok = False

    # Check Key Sections
    full_doc_text = ""
    for idx, page in enumerate(reader.pages):
        full_doc_text += page.extract_text() + "\n"

    sections_to_check = [
        "1. Problem Statement", "2. Executive Summary", "3. The Six Computing Paradigms",
        "4. Correctness Verification", "5. Experimental Environment", "6. Benchmarking",
        "7. MPI Strong Scaling", "8. MPI Communication Overlap",
        "9. GPU Profiling", "10. Hardware Energy Telemetry",
        "11. Comprehensive Six-Way", "12. Architectural Bottleneck",
        "13. When Parallelization Does Not Help", "14. Reproducibility Guide",
        "15. Limitations", "16. Conclusion", "17. Contributions"
    ]

    print("\n[CHECK 2] Report Structure (17 Sections):")
    all_sec_ok = True
    for sec in sections_to_check:
        found = sec in full_doc_text
        print(f"  - Section {sec}: {'[OK]' if found else '[FAIL]'}")
        if not found: all_sec_ok = False

    # Check Audited Key Metrics
    print("\n[CHECK 3] Key Audited Benchmark Metrics:")
    key_metrics = [
        ("Sequential 2.834 J/pass", r"2\.834\s*J/pass"),
        ("OpenMP 1.098 J/pass", r"1\.098\s*J/pass"),
        ("CUDA Optimized 0.277 J/pass", r"0\.277\s*J/pass"),
        ("Basic MPI 2.663 J/pass", r"2\.663\s*J/pass"),
        ("Opt MPI 1.970 J/pass", r"1\.970\s*J/pass"),
        ("Self-Consistent Efficiency 100.0%", r"100\.0%"),
        ("Self-Consistent Efficiency 56.0%", r"56\.0%"),
        ("Waitall Stall Time", r"Waitall"),
        ("Numerical Equivalence (0.00e+00)", r"0\.00e\+00")
    ]
    all_metrics_ok = True
    for label, pattern in key_metrics:
        found = bool(re.search(pattern, full_doc_text))
        print(f"  - {label}: {'[OK]' if found else '[FAIL]'}")
        if not found: all_metrics_ok = False

    # Check Embedded Images (Cover Logo + 4 Plots = 5 Images)
    print("\n[CHECK 4] Embedded Figures & Plots:")
    total_images = sum(len(page.images) for page in reader.pages)
    print(f"  - Total embedded images in PDF: {total_images} (Required: 5 including cover logo)")
    images_ok = (total_images >= 5)

    # Check Drive Link and Header/Footer Date/Time
    print("\n[CHECK 5] Drive Link & Clean Header/Footer Verification:")
    new_drive_link = "11teXqjJlCB-jTuv5LUcZallRTcDaSzYy"
    old_drive_link = "1i3eLAEA6jXPNDnx8pnbJffigIk-LwEPE"
    
    new_link_present = new_drive_link in full_doc_text
    old_link_absent = old_drive_link not in full_doc_text
    
    # Check for auto-generated browser header/footer date-time like '10/6/26, 1:31 AM'
    datetime_matches = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)', full_doc_text)
    header_footer_date_absent = (len(datetime_matches) == 0)
    
    # Check MPI timing correction text
    mpi_correction_text = "Likewise, the parallel MPI execution timings recorded during the dedicated strong-scaling benchmark session"
    mpi_correction_present = mpi_correction_text in full_doc_text

    # Check Viva Voce is completely absent
    viva_absent = ("Viva Voce" not in full_doc_text) and ("Oral Defense" not in full_doc_text)

    print(f"  - New Google Drive Link ({new_drive_link}): {'[OK] PRESENT' if new_link_present else '[FAIL] MISSING'}")
    print(f"  - Old Google Drive Link ({old_drive_link}): {'[OK] ABSENT' if old_link_absent else '[FAIL] FOUND'}")
    print(f"  - Auto-generated Date/Time in Header/Footer: {'[OK] ABSENT' if header_footer_date_absent else f'[FAIL] FOUND ({datetime_matches})'}")
    print(f"  - MPI Timing Variance Correction Paragraph: {'[OK] PRESENT' if mpi_correction_present else '[FAIL] MISSING'}")
    print(f"  - Viva Voce & Oral Defense Section: {'[OK] ABSENT' if viva_absent else '[FAIL] FOUND'}")

    drive_and_header_ok = new_link_present and old_link_absent and header_footer_date_absent and mpi_correction_present and viva_absent

    # Check 6: MPI Implementation Code Listings & Comparison Table
    print("\n[CHECK 6] MPI Implementation Code Listings & Synthesis:")
    mpi_code_checks = [
        ("Listing 3.1: Domain Decomposition", "Listing 3.1: MPI initialization and contiguous domain decomposition"),
        ("Listing 3.2: Basic MPI Halo Exchange", "Listing 3.2: Basic MPI halo exchange using blocking communication"),
        ("Listing 3.3: Basic MPI Stencil Loop", "Listing 3.3: Basic MPI local moving-average computation"),
        ("Listing 3.4: Opt MPI Non-Blocking Post", "Listing 3.4: Optimized MPI non-blocking halo exchange"),
        ("Listing 3.5: Overlapped Interior Stencil", "Listing 3.5: Interior computation overlapped with communication"),
        ("Listing 3.6: Waitall & Boundary Stencil", "Listing 3.6: Completion of communication and boundary computation"),
        ("MPI Code Comparison Table", "Data distribution"),
        ("Scatterv / Gatherv Vector Collectives", "MPI_Scatterv"),
        ("Sendrecv / PROC_NULL Blocking Exchange", "MPI_Sendrecv"),
        ("Irecv / Isend Non-Blocking Primitives", "MPI_Irecv"),
        ("Waitall Synchronization Primitive", "MPI_Waitall")
    ]
    all_mpi_code_ok = True
    for label, pat in mpi_code_checks:
        found = pat in full_doc_text
        print(f"  - {label}: {'[OK] PRESENT' if found else '[FAIL] MISSING'}")
        if not found: all_mpi_code_ok = False

    print("\n=======================================================")
    overall = all_meta_ok and all_sec_ok and all_metrics_ok and images_ok and drive_and_header_ok and all_mpi_code_ok
    print(f" OVERALL PA2 PDF VERIFICATION STATUS: {'SUCCESS [PASSED]' if overall else 'FAILED'}")
    print("=======================================================\n")
    return total_pages, file_size_mb

if __name__ == '__main__':
    read_and_convert_md()
    server, server_port = start_local_server(PROJECT_ROOT)
    print(f"Local HTTP server running on http://127.0.0.1:{server_port}")
    try:
        generate_pdf(server_port)
        verify_pdf(PDF_PATH_PA2)
    finally:
        server.shutdown()
