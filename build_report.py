"""Build report/draft.docx from report/draft.md with pandoc (pypandoc_binary),
using report/reference.docx as the style template, and report word counts.

    python build_report.py            # render draft.md from the template, count words, build the docx
    python build_report.py --counts   # render draft.md and count words; no docx

Every build overwrites report/draft.md, report/draft.docx and report/reference.docx (the styles live
in make_reference_docx). A draft.docx changed since the last build, for example edited in Word, is
first moved to report/draft.edited-<date-time>.docx, so finish the submission copy in a separate file.

Word counts (results/word_counts.json):
  standard : headings, body text, snippet captions and in-text citations of the five Parts
             (excludes title page, declaration, contents, Introduction, figure and table captions,
             tables, code, references, appendices)
  strict   : the same but including figure and table captions, tables and code snippets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
MD = REPORT / "draft.md"
DOCX = REPORT / "draft.docx"
REF = REPORT / "reference.docx"
STAMP = REPORT / "draft.docx.sha256"  # hash of draft.docx as last built, to notice later hand edits

PART_HEADINGS = ["Part 1", "Part 2", "Part 3", "Part 4", "Part 5"]
# Title-page and declaration placeholders the student fills in; the build lists any that remain
SUBMISSION_TODO = {"[name]": "student name", "[ID]": "student ID", "[repository link]": "repository link",
                   "[Signed AI declaration": "signed AI declaration"}
# Table column widths in units of 4.5 pt, the width of a typical 9-pt Calibri character. Word draws a
# full-width table 457 pt wide (the 451.3-pt A4 text width plus the outer cell margins), and each cell
# loses 5.7 pt to its left and right margins (set in make_reference_docx). SPARE covers text_width's
# largest underestimate of a real word (about 6.5%)
TEXT_UNITS, CELL_PAD, SPARE = 457 / 4.5, 5.7 / 4.5, 1.08
NARROW, WIDE = set("fijlrtIJ.,;:'!()[]|-  "), set("mwMW%")


def text_width(s: str) -> float:
    """Approximate width of 9-pt Calibri text in 4.5-pt units: narrow characters take about 0.6 of a
    typical one, capitals 1.2 and m, w and % 1.6."""
    return sum(0.6 if c in NARROW else 1.6 if c in WIDE else 1.2 if c.isupper() else 1.0 for c in s)


def make_reference_docx() -> None:
    """Create a clean reference.docx: A4 with 2.54 cm margins, 1.27 cm header and footer distances and
    a centred page number in the footer from page 2, UK English, 11.5 pt Calibri body with 1.15 line
    spacing, 9 pt single-spaced table text with 0.1 cm cell margins, and one caption look (9.5 pt,
    upright; the template bolds the label). Pandoc styles every table-cell paragraph as Compact, and a
    paragraph style overrides the table style, so the 9 pt is set on Compact (the report has no lists).
    The contents heading takes Heading 1's look, and 'toc 1' styles the stored contents entries."""
    import pypandoc
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    tmp = REPORT / "_pandoc_default_reference.docx"
    data = pypandoc.get_pandoc_path()
    subprocess.run([data, "-o", str(tmp), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(tmp))
    styles = doc.styles
    for lang in styles.element.iter(qn("w:lang")):  # docDefaults: proof the text as UK English
        lang.set(qn("w:val"), "en-GB")
    for name in ["Normal", "Body Text", "First Paragraph"]:
        if name in [s.name for s in styles]:
            st = styles[name]
            st.font.name = "Calibri"
            st.font.size = Pt(11.5)
            pf = st.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.15
            pf.space_after = Pt(6)
    if "Compact" in [s.name for s in styles]:
        st = styles["Compact"]
        st.font.name = "Calibri"
        st.font.size = Pt(9)
        pf = st.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
    for lvl, size in [(1, 15), (2, 13), (3, 12)]:
        name = f"Heading {lvl}"
        if name in [s.name for s in styles]:
            st = styles[name]
            st.font.name = "Calibri"
            st.font.size = Pt(size)
            st.font.bold = True
            st.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
            st.paragraph_format.space_after = Pt(4)
    if "TOC Heading" in [s.name for s in styles]:  # drop its own font, weight and colour: inherit Heading 1's
        rpr = styles["TOC Heading"].element.rPr
        for tag in ("w:rFonts", "w:b", "w:bCs", "w:color"):
            for el in (rpr.findall(qn(tag)) if rpr is not None else []):
                rpr.remove(el)
    for name in ["Image Caption", "Table Caption", "Caption"]:  # captions below figures and code, above tables
        if name in [s.name for s in styles]:
            st = styles[name]
            st.font.size = Pt(9.5)
            st.font.italic = False
            pf = st.paragraph_format
            pf.space_before, pf.space_after = (Pt(6), Pt(3)) if name == "Table Caption" else (Pt(3), Pt(9))
            pf.keep_together = True  # a caption never splits across pages
    if "toc 1" not in [s.name for s in styles]:  # the stored contents entries refer to style id "TOC1"
        st = styles.add_style("toc 1", WD_STYLE_TYPE.PARAGRAPH, builtin=True)
        st.element.set(qn("w:styleId"), "TOC1")
        st.base_style = styles["Normal"]
        st.paragraph_format.space_after = Pt(4)
    for name in ["Table", "Table Grid"]:
        if name in [s.name for s in styles]:
            st = styles[name]
            st.font.size = Pt(9)
            st.paragraph_format.space_after = Pt(0)
            st.paragraph_format.line_spacing = 1.0
    for mar in styles["Table"].element.iter(qn("w:tblCellMar")):  # pandoc's table style
        for side in mar.findall(qn("w:left")) + mar.findall(qn("w:right")):
            side.set(qn("w:w"), "57")  # 0.1 cm, so wide tables keep their longest words whole
    for name in ["Source Code", "Verbatim Char"]:
        if name in [s.name for s in styles]:
            st = styles[name]
            st.font.name = "Consolas"
            st.font.size = Pt(8.5)
    if "Footer" not in [s.name for s in styles]:  # the footer paragraph refers to style id "Footer"
        st = styles.add_style("Footer", WD_STYLE_TYPE.PARAGRAPH, builtin=True)
        st.element.set(qn("w:styleId"), "Footer")
        st.base_style = styles["Normal"]
        st.paragraph_format.space_after = Pt(0)
    sec = doc.sections[0]  # pandoc copies page size, margins and footer from the reference document
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Cm(2.54)
    sec.header_distance = sec.footer_distance = Cm(1.27)
    sec.gutter = Cm(0)
    sec.different_first_page_header_footer = True  # no page number on the title page
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run = OxmlElement("w:r"); txt = OxmlElement("w:t"); txt.text = "1"; run.append(txt); fld.append(run)
    fp._p.append(fld)
    doc.save(str(REF))
    tmp.unlink(missing_ok=True)


def strip_for_count(text: str, strict: bool) -> dict[str, int]:
    """Count words per Part. Non-strict: drop code blocks, tables, captions,
    figure lines. Strict: keep captions, tables and code."""
    # Cut everything before Part 1 and from References onwards
    m = re.search(r"^#+\s*Part 1\b", text, flags=re.M)
    start = m.start() if m else 0
    m2 = re.search(r"^#+\s*(References|Reference list)\b", text, flags=re.M)
    end = m2.start() if m2 else len(text)
    body = text[start:end]
    # Split by Part headings
    parts = re.split(r"(?m)^(#+\s*Part \d[^\n]*)$", body)
    counts = {}
    current = None
    for chunk in parts:
        h = re.match(r"#+\s*(Part \d)", chunk)
        if h:
            current = h.group(1)
            counts[current] = counts.get(current, 0) + len(re.findall(r"\S+", re.sub(r"^#+\s*", "", chunk)))
            continue
        if current is None:
            continue
        c = re.sub(r"```\{=openxml\}.*?```", " ", chunk, flags=re.S)
        if not strict:
            c = re.sub(r"```.*?```", " ", c, flags=re.S)                  # code blocks
            c = re.sub(r"(?m)^\|.*$", " ", c)                              # table rows
            c = re.sub(r"(?m)^(\*\*?(Figure|Table) \d+[^\n]*)$", " ", c)   # captions
            c = re.sub(r"(?m)^!\[[^\]]*\]\([^)]*\)\s*$", " ", c)           # images
            c = re.sub(r"(?m)^:\s.*$", " ", c)                             # pandoc table captions
        else:
            c = re.sub(r"(?m)^!\[([^\]]*)\]\([^)]*\)\s*$", r"\1", c)     # figure captions (alt text) kept
            c = re.sub(r"(?m)^\|[-| :]+\|\s*$", " ", c)                   # table separator rows
        c = re.sub(r"<!--.*?-->", " ", c, flags=re.S)
        c = c.replace("|", " ")
        counts[current] += len(re.findall(r"\S+", c))  # Word-style: every whitespace-separated token counts
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def word_counts() -> dict:
    text = MD.read_text(encoding="utf-8")
    std = strip_for_count(text, strict=False)
    strict = strip_for_count(text, strict=True)
    budget = {"Part 1": 400, "Part 2": 500, "Part 3": 700, "Part 4": 1000, "Part 5": 400, "total": 3000}
    out = {"standard": std, "strict": strict, "budget": budget,
           "within_5pct": {k: abs(std.get(k, 0) - v) <= 0.05 * v for k, v in budget.items() if k != "total"},
           "total_ok": std.get("total", 0) <= 3000}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "word_counts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def toc_field(md: str) -> str:
    """The contents field, with a stored result that lists the level-1 headings (no page numbers).
    Word recomputes the field when the file is opened (w:dirty) and adds page numbers; viewers that
    only show stored results (LibreOffice, Google Docs, Turnitin's viewer) show this list."""
    heads, fence = [], None
    for line in md.split("\n"):
        if line.startswith(("```", "~~~")):
            fence = None if fence == line[:3] else fence or line[:3]
        elif not fence and line.startswith("# "):
            heads.append(re.sub(r"\s*\{[^}]*\}\s*$", "", line[2:]).strip())
    if not heads:
        return ""
    runs = [f'<w:r><w:t xml:space="preserve">{escape(h)}</w:t></w:r>' for h in heads]
    runs[0] = (r'<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
               r'<w:r><w:instrText xml:space="preserve"> TOC \o "1-1" \h \z \u </w:instrText></w:r>'
               r'<w:r><w:fldChar w:fldCharType="separate"/></w:r>') + runs[0]
    runs[-1] += '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    return "".join(f'<w:p><w:pPr><w:pStyle w:val="TOC1"/></w:pPr>{r}</w:p>' for r in runs)


def size_columns(md: str) -> str:
    """Set the column widths of every pipe table that pandoc wraps. Pandoc wraps a table once any of its
    lines, the separator included, is longer than 72 characters and gives each column the share of the
    table width that its separator characters (dashes and alignment colons) have of the whole separator,
    so equal dashes split long words in narrow columns. Each column here gets at least its widest word
    with SPARE to spare, less up to a quarter unit lost to rounding the dashes (a word over ten
    characters with letters in it may break after a hyphen or dash, so only its widest piece counts; a
    no-break space joins words into one), and the width left over goes in proportion to what each column
    still needs to fit its longest cell on one line. A table whose widest words alone do not fit gets
    widths in proportion to those words, so every column is short of its widest word by the same factor:
    shorten its headers."""

    def pieces(cell):
        for word in re.split(r"[ \t]+", cell.strip()):
            yield from (re.findall(r"[^\-–]+[\-–]?", word) if len(word) > 10 and re.search("[A-Za-z]", word) else [word])

    def split_row(row):  # cells of a pipe-table row; an escaped pipe stays inside its cell
        return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]

    lines, fence = md.split("\n"), None
    sep = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")
    for i, line in enumerate(lines):
        if line.startswith(("```", "~~~")):
            fence = None if fence == line[:3] else fence or line[:3]
        if fence or not line.startswith("|") or i + 1 >= len(lines) or not sep.match(lines[i + 1]):
            continue
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        rows = [line] + lines[i + 2:j]
        if max(len(r.rstrip()) for r in rows + [lines[i + 1]]) <= 72:  # pandoc sizes such a table to its contents
            continue
        cells = [split_row(r) for r in rows]
        n = len(cells[0])
        cells = [r[:n] + [""] * (n - len(r)) for r in cells]  # pandoc pads short rows and drops extra cells
        need = [SPARE * max((text_width(t) for r in cells for t in pieces(r[k])), default=1) for k in range(n)]
        full = [max(need[k], max(text_width(r[k]) for r in cells)) for k in range(n)]
        grow, slack = [f - d for f, d in zip(full, need)], TEXT_UNITS - n * CELL_PAD - sum(need)
        if slack <= 0 or not sum(grow):
            width = need
        elif sum(grow) <= slack:
            width = [f + (slack - sum(grow)) * f / sum(full) for f in full]
        else:
            width = [d + slack * g / sum(grow) for d, g in zip(need, grow)]
        aligns = split_row(lines[i + 1])
        lines[i + 1] = "|" + "|".join(
            (":" if a.startswith(":") else "")
            + "-" * max(3, round(2 * (w + CELL_PAD)) - a.startswith(":") - a.endswith(":"))
            + (":" if a.endswith(":") else "") for a, w in zip(aligns, width)) + "|"
    return "\n".join(lines)


def render_draft() -> None:
    """Fill report/draft_template.md placeholders from results/ and write report/draft.md."""
    import json
    sys.path.insert(0, str(ROOT))
    from src.make_tables import all_tables
    run_log = json.loads((ROOT / "results" / "run_log.json").read_text(encoding="utf-8"))
    fp = json.loads((ROOT / "results" / "compute_footprint_total.json").read_text(encoding="utf-8"))
    if run_log.get("quick") or fp.get("quick_mode"):
        sys.exit("results/ hold smoke-test outputs (run_all.py --quick or LD6048_QUICK=1); "
                 "rerun python run_all.py before building the report")
    tpl = (REPORT / "draft_template.md").read_text(encoding="utf-8")
    tables = all_tables()
    for k, v in tables.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    tpl = tpl.replace("{{REFERENCES}}", (REPORT / "references.md").read_text(encoding="utf-8").strip())
    env = json.loads((ROOT / "results" / "environment.json").read_text(encoding="utf-8"))
    man = json.loads((ROOT / "results" / "data_manifest.json").read_text(encoding="utf-8"))
    pk ={p.split("==")[0].lower(): p.split("==")[1] for p in env["packages"] if "==" in p}
    appendix_a = "\n\n".join([
        "The complete pipeline is a Python package (`src/`) driven by one command, `python run_all.py`, which regenerates every "
        "analysis result and figure from the raw CSV with seed 42 and then rebuilds this report from `report/draft_template.md`. "
        "Modelling decisions, every threshold included, are made on the training split only; "
        "the hold-out test set is untouched until model selection is complete and is then used for the final evaluation and the "
        "post-selection analyses (permutation importance, SHAP, error analysis, fairness audit). Fitted pipelines, out-of-fold and test predictions are saved under `data/processed/` and `results/`. "
        "Tables 2, 3 and 5–7, the appendix table bodies and some computed values in the text are filled in from `results/` when the "
        "report is built; Tables 1 and 4, the appendix captions and the other numbers in the text were transcribed. `src/check_numbers.py` then checks "
        "every number from the Introduction to the end of Part 5, outside code and citations: each must equal, after rounding "
        "(allowing for percentages, complements, sign and monthly or annual rescaling), a value saved in `results/`, or failing "
        "that be one of a few constants from the brief and the design on its allow-list, such as a word budget or the seed, or a "
        "literature figure listed in `report/cited_numbers.json`. That is an existence check: it flags numbers that match "
        "nothing saved, but cannot tell whether a matching number describes the right quantity, and small numbers match some "
        "saved value by chance.",
        f"**Data.** `{man['file']}` (Kaggle, blastchar; obtained from IBM's public mirror of the same file), SHA-256 `{man['sha256']}`, "
        f"{man['facts']['n_rows']:,} rows × {man['facts']['n_cols']} columns, {man['facts']['n_churn_yes']:,} churners.",
        f"**Environment.** Python {env['python'].split()[0]} on {env['platform']} ({env['cpu_count']} logical cores); "
        f"scikit-learn {pk.get('scikit-learn')}, imbalanced-learn {pk.get('imbalanced-learn')}, xgboost {pk.get('xgboost')}, shap {pk.get('shap')}, "
        f"lifelines {pk.get('lifelines')}, fairlearn {pk.get('fairlearn')}, statsmodels {pk.get('statsmodels')}, pandas {pk.get('pandas')}, numpy {pk.get('numpy')}, "
        f"codecarbon {pk.get('codecarbon')} (the reference list cites the project's recommended Version 2.4.1 record); "
        f"pinned in `requirements.txt`.",
        f"**Compute.** The modelling stage (ablations, nested CV, tuning, calibration) took {fp['total_modelling_seconds'] / 60:.1f} minutes; "
        f"CodeCarbon estimated {fp['codecarbon_kg_co2e'] * 1000:.1f} g CO2e for it. Per-model timings are in `results/compute_footprint.csv`.",
        "**Layout.** `src/load.py` (manifest and fact checks), `clean.py`, `features.py` (pipelines), `eda.py`, `models.py`, `evaluate.py`, "
        "`explain.py`, `fairness.py`, `derived.py`; `results/` (JSON/CSV); `figures/` (PNG and SVG); `references/annotated_bibliography.md`.",
        "**Names used in the snippets** (defined in `src/config.py` and `src/features.py`). `ADDONS`: the six add-on service "
        "columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`); "
        "`PROTECTION`: the first four of them; `MULTILINES`: `MultipleLines`; `REFERENCE_LEVELS`: the reference category of "
        "each nominal feature (`Female`, `DSL`, `Month-to-month`, `Bank transfer (automatic)`); `numeric`: `tenure`, "
        "`MonthlyCharges`, `TotalCharges` and any engineered numeric feature; `log1p_column`: log(1 + x) applied to one "
        "column; `model_kind`: `\"linear\"` or `\"tree\"`; `build_preprocessor()`: the `ColumnTransformer` of Snippets 1 and 2; "
        "`FeatureEngineer`: a scikit-learn transformer that applies the level collapse of Snippet 2 and the selected "
        "features of `engineer_features()` (Snippet 3); `sampler`: an optional imbalanced-learn resampler.",
    ])
    tpl = tpl.replace("{{APPENDIX_A}}", appendix_a)
    der = json.loads((ROOT / "results" / "derived.json").read_text(encoding="utf-8"))
    errors = {k: v for k, v in der.items() if k.endswith("_error")}
    if errors:
        sys.exit(f"results/derived.json records errors: {errors}")
    tpl = tpl.replace("{{MODEL_MINUTES}}", f"{fp['total_modelling_seconds'] / 60:.1f}")
    tpl = tpl.replace("{{CO2E_G}}", f"{fp['codecarbon_kg_co2e'] * 1000:.1f}")
    tpl = tpl.replace("{{FIT_RATIO}}", f"{der['fit_time_ratio_rf_vs_lr']:.0f}")
    tpl = tpl.replace("{{BREAKEVEN_ALL}}", f"{der['breakeven_save_rate_all'] * 100:.0f}")
    tpl = tpl.replace("{{BREAKEVEN_S1}}", f"{der['breakeven_save_rate_S1'] * 100:.0f}")
    # Part 5 says the dependents' own, lower threshold cuts their higher miss rate at a cost in profit;
    # while the sentence's placeholders are in the template, check that the printed values still say so
    if "{{DEP_" in tpl:
        dep = {"{{DEP_MISS_BASE}}": f"{der['miss_rate_dependents'] * 100:.0f}",
               "{{NODEP_MISS}}": f"{der['miss_rate_no_dependents'] * 100:.0f}",
               "{{DEP_THR}}": f"{der['dependents_recall_parity_threshold']:.2f}",
               "{{DEP_MISS}}": f"{(1 - der['dependents_recall_parity_test_recall_dependents']) * 100:.0f}",
               "{{DEP_COST}}": f"{-der['dependents_recall_parity_profit_change']:,.0f}"}
        thresholds = json.loads((ROOT / "results" / "thresholds.json").read_text(encoding="utf-8"))
        best = json.loads((ROOT / "results" / "model_ranking.json").read_text(encoding="utf-8"))["best_single_model"]
        base, other, after = (int(dep[k]) for k in ("{{DEP_MISS_BASE}}", "{{NODEP_MISS}}", "{{DEP_MISS}}"))
        if not (other < base and after < base and float(dep["{{DEP_COST}}"].replace(",", "")) >= 1
                and float(dep["{{DEP_THR}}"]) < round(thresholds[best]["operating"], 2)):
            sys.exit("results/derived.json no longer supports Part 5's sentence on the dependents' threshold "
                     "(higher miss rate, cut by a lower threshold at a cost); reword it in report/draft_template.md")
        for k, v in dep.items():
            tpl = tpl.replace(k, v)
    tpl = tpl.replace("{{TOC}}", toc_field(tpl))
    tpl = size_columns(tpl)
    left = sorted(set(re.findall(r"\{\{[^{}\n]+\}\}", tpl)) - {"{{WORDCOUNT}}"})
    if left:
        sys.exit(f"Unfilled placeholders in report/draft_template.md: {', '.join(left)}")
    (REPORT / "draft.md").write_text(tpl, encoding="utf-8")


def fill_wordcount() -> None:
    wc = word_counts()
    s = MD.read_text(encoding="utf-8").replace("{{WORDCOUNT}}", f"{wc['standard']['total']:,}")
    MD.write_text(s, encoding="utf-8")


def fix_pagination(path: Path) -> None:
    """Pagination and caption styles that pandoc cannot express: give each table caption the Table Caption
    style and keep it on the page where its table starts, never break a row across pages, keep each of
    the main-body Tables 1-7 (5a and 5b) on one page and the last two rows of every other table together, leave a
    gap between a table and a figure that follows it, and keep each code block (a single paragraph)
    whole and on the page of the snippet caption below it, which gets the figure-caption style. The
    reference document has no Source Code style (pandoc adds its own), so code paragraphs are set here."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    from docx.text.paragraph import Paragraph
    doc = Document(str(path))
    for tbl in doc.element.body.iter(qn("w:tbl")):
        prev, nxt = tbl.getprevious(), tbl.getnext()
        caption = Paragraph(prev, doc._body) if prev is not None and prev.tag == qn("w:p") else None
        if caption is not None:
            if re.match(r"Table [A-Z]?\d+[a-z]?\.", caption.text):
                caption.style = doc.styles["Table Caption"]
            caption.paragraph_format.keep_with_next = True
        main = caption is not None and re.match(r"Table \d+[ab]?\.", caption.text) is not None  # 1-7, 5a, 5b
        rows = tbl.findall(qn("w:tr"))
        for k, tr in enumerate(rows):
            trpr = tr.get_or_add_trPr()
            if trpr.find(qn("w:cantSplit")) is None:
                trpr.insert(0, OxmlElement("w:cantSplit"))
            if k < len(rows) - 1 and (main or k == len(rows) - 2):
                for p in tr.iter(qn("w:p")):
                    Paragraph(p, doc._body).paragraph_format.keep_with_next = True
        if nxt is not None and nxt.tag == qn("w:p"):
            after = Paragraph(nxt, doc._body)
            if after.style is not None and after.style.style_id == "CaptionedFigure":
                after.paragraph_format.space_before = Pt(9)
    paras = doc.paragraphs
    for k, p in enumerate(paras):
        if p.style is not None and p.style.style_id == "SourceCode":
            p.paragraph_format.keep_together = p.paragraph_format.keep_with_next = True
            if k + 1 < len(paras) and paras[k + 1].text.startswith("Snippet"):
                paras[k + 1].style = doc.styles["Image Caption"]
    doc.save(str(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_docx() -> None:
    """Build into a temporary file, then put it in place of draft.docx. A draft.docx that differs from
    the last build (or has no build record) is first moved to draft.edited-<date-time>.docx; if Word
    holds the file open, the build stops before anything is moved or replaced."""
    import pypandoc
    make_reference_docx()  # always regenerate, so style and page-layout changes in code take effect
    tmp = REPORT / "draft.tmp.docx"
    extra = ["--reference-doc", str(REF), "--resource-path", str(ROOT),
             "--from", "markdown+pipe_tables+implicit_figures+raw_tex-smart-tex_math_dollars", "--to", "docx"]
    try:
        try:
            pypandoc.convert_file(str(MD), "docx", outputfile=str(tmp), extra_args=extra)
        except RuntimeError as e:
            sys.exit(f"pandoc could not build {tmp}: {e}")
        fix_pagination(tmp)
        try:
            if DOCX.exists():
                built = STAMP.read_text(encoding="utf-8").strip() if STAMP.exists() else None
                if built != sha256(DOCX):
                    keep = REPORT / f"draft.edited-{time.strftime('%Y%m%d-%H%M%S')}.docx"
                    os.replace(DOCX, keep)
                    print(f"report/draft.docx {'has no build record' if built is None else 'was changed after the last build'}; "
                          f"moved it to report/{keep.name}")
            os.replace(tmp, DOCX)
        except PermissionError:
            sys.exit(f"could not replace {DOCX}: close it in Word and build again")
    finally:
        tmp.unlink(missing_ok=True)
    STAMP.write_text(sha256(DOCX), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--reference-only", action="store_true")
    a = ap.parse_args()
    if a.reference_only:
        make_reference_docx(); print("wrote", REF); return
    sys.path.insert(0, str(ROOT))
    from src.config import OUT_ROOT, QUICK
    if QUICK or OUT_ROOT.resolve() != ROOT.resolve():  # tables would come from elsewhere than the text's numbers
        sys.exit("build_report.py builds the report from this repository's own results/: unset LD6048_QUICK and LD6048_OUT_ROOT")
    if (REPORT / "draft_template.md").exists():
        render_draft()
        fill_wordcount()
    if not MD.exists():
        sys.exit(f"{MD} not found")
    wc = word_counts()
    print(json.dumps(wc, indent=2))
    if not a.counts:
        build_docx()
        print("wrote", DOCX)
    todo = [what for mark, what in SUBMISSION_TODO.items() if mark in MD.read_text(encoding="utf-8")]
    if todo:
        print("Before submission, fill in (report/draft_template.md or the submission copy):", ", ".join(todo))


if __name__ == "__main__":
    main()
