"""
Extract supplementary training data from:
1. legaldoc.pdf (Indian legal templates explanatory prose) → 6 classes
2. MILPaC_IP_dataset.xlsx → label 7 (IP)
3. MILPaC_CCI_FAQ_dataset.xlsx → label 3 (Corporate)

Outputs: datasets/supplementary_labeled.csv
"""
import os
import re
import pandas as pd
from pdfminer.high_level import extract_text

LABEL_MAP = {
    0: "Contract Law", 1: "Criminal Law", 2: "Constitutional Law",
    3: "Corporate / Company Law", 4: "Property / Real Estate Law",
    5: "Family Law", 6: "Labour & Employment Law",
    7: "Intellectual Property Law", 8: "Taxation Law",
    9: "Civil Procedure / Other"
}

# UPDATE THESE PATHS to where you put the source files
PDF_PATH = "datasets/legaldoc.pdf"  # copy from /mnt/user-data/uploads/legaldoc.pdf
MILPAC_IP = "datasets/MILPaC_IP_dataset.xlsx"
MILPAC_CCI = "datasets/MILPaC_CCI_FAQ_dataset.xlsx"

all_rows = []  # list of dicts: {text, label, label_name, source}

# =============================================================
# 1. EXTRACT FROM legaldoc.pdf
# =============================================================
print("=== 1. Extracting from legaldoc.pdf ===")

DOC_LABEL_KEYWORDS = [
    # Most specific first
    ('cancellation of power of attorney', 9),
    ('power of attorney', 9),
    ('anticipatory bail', 1),
    ('legal notice for defamation', 1),
    ('defamation', 1),
    ('recovery of money', 9),
    ('legal notice for property partition', 4),
    ('property partition', 4),
    ('partition', 4),
    ('builder for delay', 4),
    ('delay in handing over', 4),
    ('lease deed', 4),
    ('rent agreement', 4),
    ('family trust', 5),
    ('separation agreement', 5),
    ('deed of adoption', 5),
    ('adoption', 5),
    ('family settlement', 5),
    ('non-disclosure', 0),
    ('confidential information', 0),
    ('business services agreement', 0),
    ('service agreement', 0),
    ('loan agreement', 0),
    ('trade mark owner', 7),
    ('trademark', 7),
    ('trade mark', 7),
    ('licence to use copyright', 7),
    ('copyright', 7),
]

KEEP_PREFIXES = [
    'What is ', 'Why is ', 'What should ',
    'Documents Required for', 'Procedure for',
    'Legal Considerations for', 'How can a lawyer'
]
SKIP_PREFIXES = ['Format for ', 'DRAFT OF ', 'Draft of ']

def label_for_header(header):
    h = header.lower()
    for keyword, label in DOC_LABEL_KEYWORDS:
        if keyword in h:
            return label
    return None

if not os.path.exists(PDF_PATH):
    print(f"  ! Skipping PDF — file not found at {PDF_PATH}")
else:
    txt = extract_text(PDF_PATH)
    lines = txt.split('\n')
    sections = []
    current = {'header': None, 'lines': [], 'skip': False}
    
    for line in lines:
        s = line.strip()
        is_keep = any(s.startswith(p) for p in KEEP_PREFIXES)
        is_skip = any(s.startswith(p) for p in SKIP_PREFIXES)
        if is_keep or is_skip:
            if current['header']:
                sections.append(current)
            current = {'header': s, 'lines': [], 'skip': is_skip}
        else:
            current['lines'].append(line)
    if current['header']:
        sections.append(current)
    
    pdf_count = 0
    for sec in sections:
        if sec['skip']:
            continue
        label = label_for_header(sec['header'])
        if label is None:
            continue
        body = '\n'.join(sec['lines']).strip()
        body = re.sub(r'Download Word Doc', '', body, flags=re.IGNORECASE)
        paragraphs = re.split(r'\n\s*\n', body)
        for para in paragraphs:
            para = re.sub(r'\s+', ' ', para).strip()
            if len(para.split()) < 40:
                continue
            if para.count('_') > 3 or para.count('....') > 2:
                continue
            if re.match(r'^\d+\.\s', para):
                continue
            if para.startswith('o ') and 'o ' in para[10:30]:
                continue
            all_rows.append({
                'text': para, 'label': label,
                'label_name': LABEL_MAP[label], 'source': 'pdf'
            })
            pdf_count += 1
    print(f"  Extracted {pdf_count} examples from PDF")

# =============================================================
# 2. EXTRACT FROM MILPaC IP (label 7)
# =============================================================
print("\n=== 2. Extracting from MILPaC IP ===")
if not os.path.exists(MILPAC_IP):
    print(f"  ! Skipping — file not found at {MILPAC_IP}")
else:
    df = pd.read_excel(MILPAC_IP)
    en = df[df['src_lang'] == 'EN'].drop_duplicates(subset=['src']).copy()
    # Only Answers (longer prose), not Questions
    en['is_answer'] = en['id'].astype(str).str.startswith('A')
    answers = en[en['is_answer']].copy()
    answers['word_count'] = answers['src'].astype(str).str.split().str.len()
    keep = answers[answers['word_count'] >= 30]
    
    ip_count = 0
    for _, row in keep.iterrows():
        text = re.sub(r'\s+', ' ', str(row['src'])).strip()
        all_rows.append({
            'text': text, 'label': 7,
            'label_name': LABEL_MAP[7], 'source': 'milpac_ip'
        })
        ip_count += 1
    print(f"  Extracted {ip_count} IP examples from MILPaC IP")

# =============================================================
# 3. EXTRACT FROM MILPaC CCI (label 3 — Corporate)
# =============================================================
print("\n=== 3. Extracting from MILPaC CCI FAQ ===")

# Filter procedural form-filling content out
SKIP_PATTERNS = [
    'arial 12', 'demand draft', 'banker', 'electronic clearance',
    'pen drive', 'cd/', 'a4 size', 'photocopy',
]
# Keep substantive Corporate Law content
KEEP_KEYWORDS = [
    'combination', 'cartel', 'acquisition', 'merger', 'aaec', 'enterprise',
    'competition act', 'anti-competitive', 'abuse of dominance', 'turnover',
    'penalty', 'lesser penalty', 'commission', 'ibc', 'insolvency',
    'agreement', 'vertical', 'horizontal', 'market', 'investigation',
    'section', 'regulation', 'inquiry', 'group',
]

def is_corporate_substantive(text):
    t = str(text).lower()
    if any(skip in t for skip in SKIP_PATTERNS):
        return False
    return any(kw in t for kw in KEEP_KEYWORDS)

if not os.path.exists(MILPAC_CCI):
    print(f"  ! Skipping — file not found at {MILPAC_CCI}")
else:
    df = pd.read_excel(MILPAC_CCI)
    en = df[df['src_lang'] == 'EN'].drop_duplicates(subset=['src']).copy()
    en['is_answer'] = en['id'].astype(str).str.startswith('A')
    answers = en[en['is_answer']].copy()
    answers['word_count'] = answers['src'].astype(str).str.split().str.len()
    answers['keep'] = answers['src'].apply(is_corporate_substantive)
    keep = answers[(answers['keep']) & (answers['word_count'] >= 30) & (answers['word_count'] <= 250)]
    
    cci_count = 0
    for _, row in keep.iterrows():
        text = re.sub(r'\s+', ' ', str(row['src'])).strip()
        all_rows.append({
            'text': text, 'label': 3,
            'label_name': LABEL_MAP[3], 'source': 'milpac_cci'
        })
        cci_count += 1
    print(f"  Extracted {cci_count} Corporate examples from MILPaC CCI")

# =============================================================
# 4. WRITE OUTPUT
# =============================================================
out_df = pd.DataFrame(all_rows)
print(f"\n=== TOTAL ===")
print(f"All supplementary examples: {len(out_df)}")
print(f"\nClass distribution:")
print(out_df.groupby(['label', 'label_name', 'source']).size().to_string())

out_path = "datasets/supplementary_labeled.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
