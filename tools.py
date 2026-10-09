"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import re
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Retries on: RetryableError, HTTP 429/500/502/503/504, httpx.TransportError.
    Uses Retry-After header if present, else exponential backoff with jitter.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            delay = e.retry_after if e.retry_after is not None else min(base * (2 ** attempt), cap)
            # add jitter
            import random
            delay = delay * (0.5 + random.random())
            time.sleep(delay)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504):
                if attempt == attempts - 1:
                    raise
                # try to read Retry-After header
                retry_after = e.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(base * (2 ** attempt), cap)
                import random
                delay = delay * (0.5 + random.random())
                time.sleep(delay)
            else:
                raise
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            delay = min(base * (2 ** attempt), cap)
            import random
            delay = delay * (0.5 + random.random())
            time.sleep(delay)


# Track last arXiv call time for etiquette
_last_arxiv_call = 0.0


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_call

    # Sanitize query: keep only word characters
    terms = re.findall(r'\w+', query)
    if not terms:
        return "NO RESULTS"

    # Clamp max_results
    max_results = max(1, min(30, max_results))

    def _call():
        global _last_arxiv_call
        # Respect arXiv etiquette: at least 3 seconds between calls
        elapsed = time.time() - _last_arxiv_call
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)

        params = {
            "search_query": " AND ".join(f"all:{t}" for t in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
        resp = httpx.get(ARXIV_URL, params=params, timeout=30)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RetryableError("arXiv rate limited", retry_after=float(retry_after) if retry_after else None)
        if resp.status_code >= 500:
            raise RetryableError(f"arXiv server error {resp.status_code}")
        resp.raise_for_status()
        _last_arxiv_call = time.time()
        return resp.text

    try:
        xml_text = with_retry(_call, attempts=5, base=2.0, cap=60.0)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    try:
        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            atom_id = entry.find("atom:id", ns)
            published = entry.find("atom:published", ns)
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)

            raw_id = atom_id.text.rsplit("/", 1)[-1] if atom_id is not None and atom_id.text else ""
            # Remove version suffix
            vid = re.sub(r'v\d+$', '', raw_id)

            url = f"https://arxiv.org/abs/{vid}" if vid else ""
            date = published.text[:10] if published is not None and published.text else ""
            title = re.sub(r'\s+', ' ', title_el.text.strip()) if title_el is not None and title_el.text else ""
            summary = re.sub(r'\s+', ' ', summary_el.text.strip()) if summary_el is not None and summary_el.text else ""
            summary = summary[:600]

            if vid:
                records.append({
                    "id": vid,
                    "url": url,
                    "published": date,
                    "title": title,
                    "summary": summary,
                })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    limit = max(1, min(100, limit))
    params = {"limit": limit}
    if date:
        params["date"] = date

    def _call():
        resp = httpx.get(HF_DAILY_URL, params=params, timeout=30)
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"HF daily error {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    try:
        data = with_retry(_call, attempts=5, base=1.0, cap=30.0)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    try:
        records = []
        for item in data:
            paper = item.get("paper", {})
            pid = paper.get("id")
            if not pid:
                continue

            title = paper.get("title", "")
            summary = paper.get("summary", "")
            if keyword:
                kw = keyword.lower()
                if kw not in title.lower() and kw not in summary.lower():
                    continue

            upvotes = paper.get("upvotes", 0)
            github = paper.get("githubRepo", "")
            stars = paper.get("githubStars", 0) or 0
            pub_date = paper.get("publishedAt", "")[:10]

            records.append({
                "id": pid,
                "url": f"https://huggingface.co/papers/{pid}",
                "published": pub_date,
                "title": title,
                "summary": summary[:600],
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        # Sort by upvotes descending
        records.sort(key=lambda x: x["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    limit = max(1, min(50, limit))

    def _call():
        resp = httpx.get(HF_SEARCH_URL, params={"q": query, "limit": limit}, timeout=30)
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"HF search error {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    try:
        data = with_retry(_call, attempts=5, base=1.0, cap=30.0)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    try:
        records = []
        for paper in data:
            pid = paper.get("id")
            if not pid:
                continue

            # Prefer ai_summary over summary
            summary = paper.get("ai_summary") or paper.get("summary", "")
            upvotes = paper.get("upvotes", 0)
            github = paper.get("githubRepo", "")
            stars = paper.get("githubStars", 0) or 0
            pub_date = paper.get("publishedAt", "")[:10]

            records.append({
                "id": pid,
                "url": f"https://huggingface.co/papers/{pid}",
                "published": pub_date,
                "title": paper.get("title", ""),
                "summary": summary[:600],
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    if not objective:
        objective = f"Find information about: {query}"

    exa_key = os.getenv("EXA_API_KEY", "")
    url = EXA_URL
    if exa_key:
        url = f"{EXA_URL}?exaApiKey={exa_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {
                "query": query,
                "objective": objective,
                "numResults": max(1, min(20, num_results)),
            }
        }
    }

    def _call():
        resp = httpx.post(url, json=payload, timeout=60,
                          headers={"Content-Type": "application/json",
                                   "Accept": "application/json, text/event-stream"})
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"Exa error {resp.status_code}")
        resp.raise_for_status()

        # Parse SSE
        text = resp.text
        for line in text.split("\n"):
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:])
                    # Check for Exa rate limit flag
                    if "result" in data:
                        meta = data.get("result", {}).get("_meta", {})
                        if meta.get("ratelimit", False):
                            raise RetryableError("Exa rate limited")
                        return data
                    if "error" in data:
                        err = data["error"]
                        raise RetryableError(f"Exa error: {err.get('message', err)}")
                except json.JSONDecodeError:
                    pass
        raise RetryableError("No valid response from Exa")

    try:
        result = with_retry(_call, attempts=5, base=2.0, cap=60.0)
    except Exception as e:
        # Redact API key from error message
        err_msg = str(e)
        if exa_key:
            err_msg = err_msg.replace(exa_key, "[EXA_KEY]")
        return f"ERROR: {type(e).__name__}: {err_msg}"

    try:
        content_items = result.get("result", {}).get("content", [])
        texts = []
        for item in content_items:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        output = "\n\n---\n\n".join(texts)

        # Check for rate limit message in output
        if "rate limit" in output.lower() or "exceeds your quota" in output.lower():
            return f"ERROR: RetryableError: Exa rate limit detected in content"

        return output if output else "NO RESULTS"
    except Exception as e:
        err_msg = str(e)
        if exa_key:
            err_msg = err_msg.replace(exa_key, "[EXA_KEY]")
        return f"ERROR: {type(e).__name__}: {err_msg}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    exa_key = os.getenv("EXA_API_KEY", "")
    req_url = EXA_URL
    if exa_key:
        req_url = f"{EXA_URL}?exaApiKey={exa_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_fetch_exa",
            "arguments": {"urls": [url]},
        }
    }

    def _call():
        resp = httpx.post(req_url, json=payload, timeout=60,
                          headers={"Content-Type": "application/json",
                                   "Accept": "application/json, text/event-stream"})
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"Exa error {resp.status_code}")
        resp.raise_for_status()

        text = resp.text
        for line in text.split("\n"):
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:])
                    if "result" in data:
                        meta = data.get("result", {}).get("_meta", {})
                        if meta.get("ratelimit", False):
                            raise RetryableError("Exa rate limited")
                        return data
                    if "error" in data:
                        err = data["error"]
                        raise RetryableError(f"Exa error: {err.get('message', err)}")
                except json.JSONDecodeError:
                    pass
        raise RetryableError("No valid response from Exa")

    try:
        result = with_retry(_call, attempts=5, base=2.0, cap=60.0)
    except Exception as e:
        err_msg = str(e)
        if exa_key:
            err_msg = err_msg.replace(exa_key, "[EXA_KEY]")
        return f"ERROR: {type(e).__name__}: {err_msg}"

    try:
        content_items = result.get("result", {}).get("content", [])
        texts = []
        for item in content_items:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        output = "\n\n".join(texts)
        output = output[:12000]
        return output if output else "NO RESULTS"
    except Exception as e:
        err_msg = str(e)
        if exa_key:
            err_msg = err_msg.replace(exa_key, "[EXA_KEY]")
        return f"ERROR: {type(e).__name__}: {err_msg}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
