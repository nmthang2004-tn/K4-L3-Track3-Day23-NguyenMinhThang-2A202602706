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
    """Return a list of problem strings (empty list = OK)."""
    problems = []

    # 1. sources must not be empty
    if not sources:
        return ["no sources in sources.json"]

    # 2. validate each source entry
    seen_urls = set()
    source_numbers = set()
    for entry in sources:
        n = entry.get("n")
        url = entry.get("url", "")

        # n must be an int
        if not isinstance(n, int):
            problems.append(f"source has non-integer n: {n!r}")

        # url must start with http:// or https://
        if not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}] has invalid url: {url!r}")

        # url must not be duplicated
        if url in seen_urls:
            problems.append(f"duplicate url: {url}")
        seen_urls.add(url)

        if isinstance(n, int):
            source_numbers.add(n)

    # 3. split report_text at "## References"
    parts = re.split(r"(?m)^##[ \t]+References[ \t]*$", report_text, maxsplit=1)
    body = parts[0]
    if len(parts) == 1:
        problems.append("missing '## References' heading")

    # 4. cited = set of numbers found as [n] in the BODY only
    #    ignore code spans and Markdown links [n](url)
    cited = set()
    body_parts = re.split(r"(```.*?```|`[^`\n]*`|\[[\d,\-– ]+\]\([^)]+\))", body, flags=re.DOTALL)
    for part in body_parts:
        # skip code/markdown-link parts (odd indexes)
        if re.match(r"```.*?```|`[^`\n]*`|\[[\d,\-– ]+\]\([^)]+\)", part, re.DOTALL):
            continue
        for m in re.finditer(r"\[(\d+)\](?!\()", part):
            cited.add(int(m.group(1)))

    # every [n] in body must exist in sources
    for n in sorted(cited):
        if n not in source_numbers:
            problems.append(f"[{n}] cited but missing from sources.json")

    # every source number must be cited
    for n in sorted(source_numbers):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # 5. validate References section
    if len(parts) >= 2:
        ref_section = parts[1]
        ref_lines = [line for line in ref_section.split("\n") if re.match(r"^\s*\[\d+\]", line)]

        # each source needs exactly one reference line
        ref_line_nums = set()
        for line in ref_lines:
            m = re.match(r"^\s*\[(\d+)\]", line)
            if m:
                n = int(m.group(1))
                if n in ref_line_nums:
                    problems.append(f"reference line has duplicate number [{n}]")
                ref_line_nums.add(n)

        # check for missing or extra numbers
        for n in sorted(source_numbers):
            if n not in ref_line_nums:
                problems.append(f"source [{n}] has no reference line")

        for n in sorted(ref_line_nums):
            if n not in source_numbers:
                problems.append(f"reference [{n}] is not a source")

        # each reference line must have exactly one http(s) URL matching the source
        for line in ref_lines:
            m = re.match(r"^\s*\[(\d+)\]", line)
            if m:
                n = int(m.group(1))
                urls_in_line = re.findall(r"https?://\S+", line)
                if len(urls_in_line) != 1:
                    problems.append(f"reference [{n}] line must have exactly one URL, found {len(urls_in_line)}")
                elif n in source_numbers:
                    expected_url = next((s.get("url") for s in sources if s.get("n") == n), None)
                    if expected_url and urls_in_line[0] != expected_url:
                        problems.append(f"reference [{n}] URL mismatch: {urls_in_line[0]} != {expected_url}")

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
