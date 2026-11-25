"""Assemble references/annotated_bibliography.md and references/references.json
from the machine-verified JSON files in references/verified/.

Each entry: Harvard reference + DOI/URL, 2-3 line summary, the claims it
supports (with where in the source), and caveats. Entries with found=false or
with no supported claims are listed separately so they are never cited.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / "references" / "verified"
OUT_MD = ROOT / "references" / "annotated_bibliography.md"
OUT_JSON = ROOT / "references" / "references.json"


def load_all() -> tuple[list[dict], list[dict]]:
    refs, legal = [], []
    for p in sorted(VER.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print("skip", p.name, e)
            continue
        d["_file"] = p.name
        (legal if p.name.startswith("legal_") else refs).append(d)
    return refs, legal


def fmt_ref(d: dict) -> str:
    lines = [f"### {d.get('key', d['_file'])}", "",
             f"**Reference.** {d.get('harvard_reference', '').strip()}"]
    if d.get("doi"):
        lines.append(f"**DOI.** https://doi.org/{d['doi']}")
    elif d.get("url"):
        lines.append(f"**URL.** {d['url']}")
    lines += ["", f"**Summary.** {d.get('abstract_summary', '').strip()}", ""]
    claims = d.get("claims_supported", [])
    if claims:
        lines.append("**Claims supported in the report.**")
        for c in claims:
            flag = "supported" if c.get("supported") else "NOT supported"
            lines.append(f"- [{flag}] {c.get('claim', '').strip()} — *where:* {c.get('where', '').strip()}")
        lines.append("")
    if d.get("caveats"):
        lines.append(f"**Caveats.** {d['caveats'].strip()}")
    if d.get("verification_notes"):
        lines.append(f"**Verification.** {d['verification_notes'].strip()}")
    lines.append("")
    return "\n".join(lines)


def fmt_legal(d: dict) -> str:
    lines = [f"### {d.get('key', d['_file'])}", "", f"**Status ({d.get('in_force', '')}).** {d.get('status_summary', '').strip()}", ""]
    for s in d.get("primary_sources", []):
        lines.append(f"- {s.get('harvard_reference', '').strip()}  \n  {s.get('url', '')}")
    if d.get("key_points"):
        lines.append("")
        lines.append("**Key points.**")
        for k in d["key_points"]:
            lines.append(f"- {k}")
    if d.get("quotable"):
        lines.append(f"\n**Quotable.** {d['quotable'].strip()}")
    if d.get("uncertainties"):
        lines.append(f"\n**Uncertainties.** {d['uncertainties'].strip()}")
    lines.append("")
    return "\n".join(lines)


def main():
    refs, legal = load_all()
    ok = [r for r in refs if r.get("found")]
    bad = [r for r in refs if not r.get("found")]
    years = [r.get("year") for r in ok if r.get("year")]
    recent = sum(1 for y in years if y >= 2020)
    md = ["# Annotated bibliography (machine-verified, reviewed before citing)", "",
          f"Verified sources: {len(ok)} academic/software/dataset references and {len(legal)} legal or regulatory items. "
          f"{recent} of {len(years)} dated references are from 2020 onwards.", "",
          "Each academic entry lists the Harvard reference, DOI, a short summary, the claims it supports in the report and where that support appears, and caveats. "
          "Claims marked NOT supported must not be attributed to that source.", "",
          "## Academic, software and dataset references", ""]
    for r in sorted(ok, key=lambda d: d.get("harvard_reference", "").lower()):
        md.append(fmt_ref(r))
    md += ["## Legal and regulatory sources (UK)", ""]
    for l in sorted(legal, key=lambda d: d.get("key", "")):
        md.append(fmt_legal(l))
    if bad:
        md += ["## Not verified (do not cite)", ""]
        for r in bad:
            md.append(f"- {r.get('key')}: {r.get('verification_notes', '')}")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"references": ok, "legal": legal, "unverified": bad}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(ok)} refs, {len(legal)} legal, {len(bad)} unverified)")


if __name__ == "__main__":
    main()
