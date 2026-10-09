"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars."""
    if not topic:
        return "topic"
    # Remove path traversal attempts
    topic = topic.replace("..", "")
    # Replace runs of non-word chars with single dash
    slug = re.sub(r'\W+', '-', topic.lower())
    # Remove leading/trailing dashes
    slug = slug.strip('-')
    # Truncate to 60 chars
    slug = slug[:60]
    return slug if slug else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Please write a comprehensive research survey about: {topic}

Follow the workflow:
1. Use write_todos to plan your work
2. Split into N >= 3 sub-questions
3. Delegate to researcher subagents in parallel
4. Merge sources into sources.json (at least 3 of 4 families: arxiv, hf-daily, hf-search, web)
5. Write the report to {REPORT_PATH} following the template (TL;DR, Background, Themes, Trends)
6. Run finalize_citations.py to generate ## References
7. Run check_citations.py until it prints "OK"
8. Have citation-checker verify a few claims

Remember: Cite at least 3 source families. Cite Hugging Face papers. Never invent facts.
"""


def summarize(messages, elapsed, model_name):
    """Return stats dict from lead messages."""
    stats = {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": 0,
        "tool_calls": {},
        "tokens": {"input": 0, "output": 0},
    }

    for msg in messages:
        # Count subagent calls (task tool invocations)
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc.get("name") == "task":
                    stats["subagent_calls"] += 1
                name = tc.get("name", "unknown")
                stats["tool_calls"][name] = stats["tool_calls"].get(name, 0) + 1

        # Count tokens from usage_metadata
        if hasattr(msg, "additional_kwargs"):
            usage = msg.additional_kwargs.get("usage_metadata", {})
            if usage:
                stats["tokens"]["input"] += usage.get("input_tokens", 0)
                stats["tokens"]["output"] += usage.get("output_tokens", 0)

    return stats


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download report and sources from sandbox, save to reports_dir."""
    slug = slugify(topic)
    files = download(backend, [REPORT_PATH, SOURCES_PATH])

    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    # Validate we have content
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"FAILED: report.md is missing or empty in sandbox")

    if not sources_bytes:
        raise RuntimeError(f"FAILED: sources.json is missing or empty in sandbox")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(f"FAILED: sources.json is not valid JSON: {e}")

    if not sources:
        raise RuntimeError("FAILED: sources.json is empty")

    # Get source families
    source_families = sorted(set(s.get("source", "") for s in sources if s.get("source")))

    # Summarize
    stats = summarize(messages, elapsed, model_name)
    stats["topic"] = topic  # Required by self_check.py
    stats["n_sources"] = len(sources)
    stats["source_families"] = source_families

    # Build output paths
    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    # Write files
    report_path.write_bytes(report_bytes)
    sources_path.write_bytes(sources_bytes)
    meta_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    return report_path


def main(topic):
    """Return exit code: 0 ok, 1 failed, 2 no topic."""
    if not topic or not topic.strip():
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    topic = topic.strip()
    print(f"Starting research: {topic}")

    try:
        model = make_model()
    except Exception as e:
        print(f"FAILED: Could not initialize model: {e}", file=sys.stderr)
        return 1

    start = time.monotonic()
    try:
        with open_sandbox() as backend:
            # Create directories
            backend.execute("mkdir -p /tmp/work/research/notes /tmp/work/report")

            # Upload scripts to sandbox
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })

            # Build and run agent
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000}
            )

            elapsed = time.monotonic() - start
            messages = result.get("messages", [])

            # Save outputs
            report_path = save_outputs(backend, topic, messages, elapsed, os.getenv("LAB_MODEL", "unknown"))

            print(f"SUCCESS: Report saved to {report_path}")
            return 0

    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"FAILED: {e}", file=sys.stderr)
        print(f"Elapsed time: {elapsed:.1f}s", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
