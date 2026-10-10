"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    report_text = str(report_text).replace("\r\n", "\n").replace("\r", "\n")
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_number = {}
    seen_urls = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source at index {index} is not an object")
            continue
        number = source.get("n")
        url = source.get("url")
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(f"source at index {index} has non-integer n")
        elif number in by_number:
            problems.append(f"source number [{number}] is duplicated")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{number}] has invalid url")
        elif url in seen_urls:
            problems.append(f"source url is duplicated: {url}")
        else:
            seen_urls.add(url)

    heading = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
    headings = list(heading.finditer(report_text))
    if not headings:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        if len(headings) > 1:
            problems.append("multiple ## References headings")
        match = headings[-1]
        body, references = report_text[:match.start()], report_text[match.end():]

    # Ignore fenced/inline code and Markdown links. Expand [1, 2] and [1-3]
    # so the validator agrees with finalize_citations.py.
    code = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
    citation_group = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\](?!\()")

    def group_numbers(value):
        result = []
        for part in re.split(r"\s*,\s*", value):
            span = re.fullmatch(r"(\d+)\s*[\-–]\s*(\d+)", part)
            if span:
                first, last = int(span.group(1)), int(span.group(2))
                if first <= last and last - first <= 200:
                    result.extend(range(first, last + 1))
                else:
                    result.extend((first, last))
            else:
                result.append(int(part))
        return result

    cited = set()
    for segment_index, segment in enumerate(code.split(body)):
        if segment_index % 2:
            continue
        for match in citation_group.finditer(segment):
            cited.update(group_numbers(match.group(1)))

    source_numbers = set(by_number)
    for number in sorted(cited - source_numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(source_numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = {}
    reference_line = re.compile(r"^\s*\[(\d+)\](?!\()")
    url_pattern = re.compile(r"https?://[^\s<>]+")
    for line in references.splitlines():
        match = reference_line.match(line)
        if not match:
            continue
        number = int(match.group(1))
        reference_lines.setdefault(number, []).append(line)

    for number in sorted(source_numbers):
        lines = reference_lines.get(number, [])
        if not lines:
            problems.append(f"source [{number}] missing from References")
            continue
        if len(lines) > 1:
            problems.append(f"source [{number}] has multiple reference lines")
        for line in lines:
            urls = url_pattern.findall(line)
            if len(urls) != 1:
                problems.append(f"reference [{number}] must contain exactly one URL")
            elif urls[0] != by_number[number].get("url"):
                problems.append(f"reference [{number}] URL does not match sources.json")
    for number in sorted(set(reference_lines) - source_numbers):
        problems.append(f"reference [{number}] missing from sources.json")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
