"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import html
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from urllib.parse import quote, quote_plus

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            error = exc
        except httpx.TransportError as exc:
            error = RetryableError(str(exc))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS:
                raise
            retry_after = _retry_after(exc.response.headers.get("Retry-After"))
            error = RetryableError(str(exc), retry_after=retry_after)
        if attempt == attempts - 1:
            raise error
        if error.retry_after is not None:
            delay = min(float(cap), max(0.0, float(error.retry_after)))
        else:
            delay = min(float(cap), float(base) * (2 ** attempt) + random.uniform(0, max(0.0, float(base))))
        time.sleep(delay)
    raise RuntimeError("retry loop ended unexpectedly")


def _retry_after(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _check_response(response):
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(
            f"HTTP {response.status_code}",
            retry_after=_retry_after(response.headers.get("Retry-After")),
        )
    response.raise_for_status()
    return response


def _http_request(method, url, **kwargs):
    """Use configured proxies first, then bypass a broken proxy on transport failure.

    Some classroom/conda environments export stale HTTP(S)_PROXY variables. A direct
    fallback keeps public research APIs usable while still honoring a working proxy.
    This helper is always called from inside ``with_retry``.
    """
    try:
        return httpx.request(method, url, **kwargs)
    except httpx.TransportError:
        with httpx.Client(trust_env=False) as client:
            return client.request(method, url, **kwargs)


def _compact(value, limit=600):
    return " ".join(str(value or "").split())[:limit]


def _result(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


def _error(exc, secret=""):
    message = str(exc)
    if secret:
        for form in {secret, quote(secret, safe=""), quote_plus(secret)}:
            message = message.replace(form, "***REDACTED***")
    return f"ERROR: {type(exc).__name__}: {message}"


# ---- arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    try:
        terms = [term for term in re.findall(r"[\w-]+", str(query), flags=re.UNICODE)
                 if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        try:
            limit = max(1, min(30, int(max_results)))
        except (TypeError, ValueError):
            limit = 10
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": limit,
            "start": 0,
        }

        def request():
            global _ARXIV_LAST_CALL
            with _ARXIV_LOCK:
                wait = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL)
                if wait > 0:
                    time.sleep(wait)
                try:
                    response = _http_request(
                        "GET", ARXIV_URL, params=params, timeout=30.0, follow_redirects=False
                    )
                finally:
                    _ARXIV_LAST_CALL = time.monotonic()
            return _check_response(response)

        response = with_retry(request, attempts=7, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", namespace):
            raw_id = _compact(entry.findtext("atom:id", default="", namespaces=namespace))
            paper_id = raw_id.rsplit("/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _compact(entry.findtext("atom:published", default="", namespaces=namespace))[:10],
                "title": _compact(entry.findtext("atom:title", default="", namespaces=namespace)),
                "summary": _compact(entry.findtext("atom:summary", default="", namespaces=namespace), 600),
            })
        return _result(records)
    except Exception as exc:  # tools must never raise into the agent
        return _error(exc)


def arxiv_lookup_ids(paper_ids) -> str:
    """Fetch verified arXiv abstract pages by ID as a fallback when the search API is rate-limited."""
    records = []
    try:
        for raw_id in list(paper_ids)[:5]:
            paper_id = re.sub(r"v\d+$", "", str(raw_id).strip())
            if not re.fullmatch(r"\d{4}\.\d{4,5}", paper_id):
                continue
            url = f"https://arxiv.org/abs/{paper_id}"
            response = with_retry(
                lambda url=url: _check_response(_http_request("GET", url, timeout=30.0)),
                attempts=3,
                base=1.0,
                cap=10.0,
            )
            page = response.text

            def meta(name):
                match = re.search(
                    rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
                    page,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                return _compact(html.unescape(match.group(1)), 600) if match else ""

            records.append({
                "id": paper_id,
                "url": url,
                "published": meta("citation_date")[:10].replace("/", "-"),
                "title": meta("citation_title") or f"arXiv {paper_id}",
                "summary": meta("description"),
            })
        return _result(records)
    except Exception as exc:
        return _error(exc)


# ---- Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        try:
            limit = max(1, min(100, int(limit)))
        except (TypeError, ValueError):
            limit = 30
        params = {"limit": limit}
        if str(date).strip():
            params["date"] = str(date).strip()
        response = with_retry(lambda: _check_response(_http_request(
            "GET", HF_DAILY_URL, params=params, timeout=30.0
        )))
        records = _hf_records(response.json(), prefer_ai_summary=False)
        needle = str(keyword).strip().casefold()
        if needle:
            records = [record for record in records if needle in f"{record['title']} {record['summary']}".casefold()]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return _result(records)
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not str(query).strip():
            return "NO RESULTS"
        try:
            limit = max(1, min(50, int(limit)))
        except (TypeError, ValueError):
            limit = 10
        response = with_retry(lambda: _check_response(_http_request(
            "GET", HF_SEARCH_URL, params={"q": str(query).strip(), "limit": limit}, timeout=30.0
        )))
        return _result(_hf_records(response.json(), prefer_ai_summary=True))
    except Exception as exc:
        return _error(exc)


def _hf_records(payload, *, prefer_ai_summary):
    if not isinstance(payload, list):
        return []
    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        paper_id = paper.get("id")
        if not paper_id:
            continue

        def field(name, default=""):
            value = paper.get(name)
            return item.get(name, default) if value is None else value

        if prefer_ai_summary:
            summary = paper.get("ai_summary") or item.get("ai_summary") or field("summary")
        else:
            summary = field("summary")
        published = field("publishedAt") or field("published")
        upvotes = field("upvotes", 0) or 0
        stars = field("githubStars", 0) or 0
        try:
            upvotes = int(upvotes)
        except (TypeError, ValueError):
            upvotes = 0
        try:
            stars = int(stars)
        except (TypeError, ValueError):
            stars = 0
        records.append({
            "id": str(paper_id),
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": _compact(published)[:10],
            "title": _compact(field("title")),
            "summary": _compact(summary, 600),
            "upvotes": upvotes,
            "github": field("githubRepo") or "",
            "stars": stars,
        })
    return records


# ---- web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    if not str(query).strip():
        return "NO RESULTS"
    try:
        count = max(1, min(10, int(num_results)))
    except (TypeError, ValueError):
        count = 5
    objective = str(objective).strip() or f"Find reliable, relevant sources about {query}"
    return _safe_mcp_call("web_search_exa", {
        "query": str(query).strip(), "objective": objective, "numResults": count,
    })


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    if not str(url).strip():
        return "NO RESULTS"
    result = _safe_mcp_call("web_fetch_exa", {"urls": [str(url).strip()]})
    return result[:12000] if not result.startswith("ERROR:") else result


def _safe_mcp_call(name, arguments):
    secret = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        text = with_retry(lambda: _mcp_call(name, arguments, secret), attempts=7, base=2.0, cap=60.0)
        return text if text.strip() else "NO RESULTS"
    except Exception as exc:
        return _error(exc, secret)


def _mcp_call(name, arguments, secret):
    endpoint = EXA_URL
    if secret:
        endpoint = str(httpx.URL(EXA_URL, params={"exaApiKey": secret}))
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = _http_request(
        "POST", endpoint,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        json=payload,
        timeout=60.0,
    )
    _check_response(response)
    data = _decode_mcp_response(response)
    if data.get("error"):
        raise RuntimeError(f"Exa JSON-RPC error: {data['error']}")
    result = data.get("result") or {}
    parts = [entry.get("text", "") for entry in result.get("content", [])
             if isinstance(entry, dict) and entry.get("type") == "text"]
    text = "\n".join(part for part in parts if part)
    meta_text = json.dumps(result.get("_meta", {}), ensure_ascii=False).casefold()
    combined = f"{meta_text}\n{text.casefold()}"
    if any(marker in combined for marker in (
        "rate limit", "ratelimit", "rate_limit", "too many requests", "rate-limit", "limit exceeded",
    )):
        raise RetryableError("Exa rate limit", retry_after=_find_retry_after(result.get("_meta")))
    return text


def _decode_mcp_response(response):
    content_type = response.headers.get("content-type", "").casefold()
    if "text/event-stream" not in content_type:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Exa returned a non-object JSON-RPC response")
        return data
    events = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        value = json.loads(raw)
        if isinstance(value, dict):
            events.append(value)
    if not events:
        raise ValueError("Exa returned an empty event stream")
    return events[-1]


def _find_retry_after(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold().replace("_", "") in {"retryafter", "retryafterseconds"}:
                parsed = _retry_after(item)
                if parsed is not None:
                    return parsed
            parsed = _find_retry_after(item)
            if parsed is not None:
                return parsed
    elif isinstance(value, list):
        for item in value:
            parsed = _find_retry_after(item)
            if parsed is not None:
                return parsed
    return None


# ---- registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
