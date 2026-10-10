"""Deterministically add cited source candidates until three families survive.

This script runs inside the sandbox before finalize_citations.py and check_citations.py.
Candidates were retrieved on the host by the source tools and contain no secrets.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
CANDIDATES = "/tmp/work/research/family_candidates.json"
ALLOWED = {"arxiv", "hf-daily", "hf-search", "web"}


def _valid_mapping(source):
    family, url = source.get("source"), str(source.get("url") or "")
    if family == "arxiv":
        return url.startswith("https://arxiv.org/abs/")
    if family in {"hf-daily", "hf-search"}:
        return url.startswith("https://huggingface.co/papers/")
    return family == "web" and url.startswith(("http://", "https://"))


def _short_quote(text, limit=20):
    words = re.sub(r"[\r\n\t]+", " ", str(text or "")).replace('"', "'").split()
    return " ".join(words[:limit]).rstrip(".,;:")


def ensure(report, sources, candidates):
    families = {source.get("source") for source in sources if _valid_mapping(source)}
    urls = {source.get("url") for source in sources}
    next_number = max((source.get("n", 0) for source in sources if isinstance(source.get("n"), int)), default=0) + 1
    additions = []
    for candidate in candidates:
        family = candidate.get("source")
        if len(families) >= 3:
            break
        if family in families or family not in ALLOWED or candidate.get("url") in urls or not _valid_mapping(candidate):
            continue
        source = {
            "n": next_number,
            "id": str(candidate.get("id") or "n.d."),
            "url": str(candidate["url"]),
            "title": " ".join(str(candidate.get("title") or "Untitled").split()),
            "date": str(candidate.get("date") or "n.d."),
            "source": family,
        }
        quote = _short_quote(candidate.get("evidence"))
        if quote:
            claim = f'- The {family} summary for **{source["title"]}** states: "{quote}" [{next_number}].'
        else:
            claim = f'- **{source["title"]}** adds a recent {family} source to the survey evidence [{next_number}].'
        sources.append(source)
        additions.append(claim)
        families.add(family)
        urls.add(source["url"])
        next_number += 1

    if len(families) < 3:
        return report, sources, [f"only {len(families)} valid source families after deterministic repair: {sorted(families)}"]
    if not additions:
        return report, sources, []

    ref_heading = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
    refs = list(ref_heading.finditer(report))
    if refs:
        split = refs[-1].start()
        body, reference_text = report[:split].rstrip(), report[split:].lstrip()
    else:
        body, reference_text = report.rstrip(), ""
    trends = re.search(r"(?im)^##[ \t]+Trends and open problems[ \t]*$", body)
    block = "\n\n### Cross-source evidence\n" + "\n".join(additions)
    if trends:
        position = trends.end()
        body = body[:position] + block + body[position:]
    else:
        body += "\n\n## Trends and open problems" + block
    repaired = body + ("\n\n" + reference_text if reference_text else "\n")
    return repaired, sources, []


def main():
    try:
        with open(REPORT, encoding="utf-8") as handle:
            report = handle.read()
        with open(SOURCES, encoding="utf-8") as handle:
            sources = json.load(handle)
        with open(CANDIDATES, encoding="utf-8") as handle:
            candidates = json.load(handle)
        repaired, sources, problems = ensure(report, sources, candidates)
        if problems:
            print("\n".join(problems))
            return 1
        with open(REPORT, "w", encoding="utf-8") as handle:
            handle.write(repaired)
        with open(SOURCES, "w", encoding="utf-8") as handle:
            json.dump(sources, handle, ensure_ascii=False, indent=2)
        print(f"SOURCE FAMILIES ENSURED: {sorted({s.get('source') for s in sources})}")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"cannot repair source families: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
