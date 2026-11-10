"""Fetch Crossref metadata and format literature references."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def fetch_doi(doi: str):
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "XAI-Verify-LR/1.0 (mailto:research@example.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())["message"]
    except Exception as exc:
        return {"error": str(exc), "DOI": doi}


def fmt_authors(authors, max_auth=3):
    if not authors:
        return ""
    names = []
    for author in authors[:max_auth]:
        family = author.get("family", "")
        given = author.get("given", "")
        if family and given:
            initials = ". ".join(part[0] for part in given.replace(".", "").split() if part) + "."
            names.append(f"{initials} {family}")
        elif family:
            names.append(family)
    if len(authors) > max_auth:
        names.append("et al.")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def fmt_ref(index, cr, fallback_title=""):
    if not cr or "error" in cr:
        return f"[{index}] {fallback_title} (metadata unavailable)"

    title = (cr.get("title") or [""])[0]
    authors = fmt_authors(cr.get("author"))
    year = ""
    for key in ("published-print", "published-online", "created", "issued"):
        parts = cr.get(key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            year = str(parts[0][0])
            break

    journal = ""
    if cr.get("container-title"):
        journal = cr["container-title"][0]
    elif cr.get("publisher"):
        journal = cr["publisher"]

    vol = cr.get("volume", "")
    issue = cr.get("issue", "")
    page = cr.get("page", "")
    doi = cr.get("DOI", "")

    segments = []
    if authors:
        segments.append(authors)
    if year:
        segments.append(f"({year})")
    segments.append(title.rstrip(".") + ".")

    if journal:
        journal_part = f"*{journal}*"
        if vol:
            journal_part += f", {vol}"
            if issue:
                journal_part += f"({issue})"
        if page:
            journal_part += f", {page.replace('-', chr(8211))}"
        segments.append(journal_part + ".")

    if doi:
        segments.append(f"doi: https://doi.org/{doi}")

    return f"[{index}] " + " ".join(segments)


def main():
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / "docs" / "literature_refs_meta.json").read_text(encoding="utf-8"))
    refs = []
    for index, item in enumerate(meta, 1):
        cr = fetch_doi(item.get("doi", ""))
        refs.append(fmt_ref(index, cr, item.get("file", "")))
        time.sleep(0.15)

    out = root / "docs" / "literature_refs_formatted.txt"
    out.write_text("\n".join(refs), encoding="utf-8")
    print("\n".join(refs))


if __name__ == "__main__":
    main()
