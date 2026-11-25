"""
conference_fetch.py  ─  Conference paper collector
─────────────────────────────────────────────────────────────
Collects accepted papers and filters them by keywords:
  - OpenReview          (CoRL, NeurIPS, ...)      fetcher: openreview
  - CVF Open Access     (CVPR, ICCV, ...)         fetcher: cvf
  - RSS proceedings     (roboticsproceedings.org) fetcher: rss

Returns raw dict format compatible with paper_radar.py's Paper model.
"""

import re


def _get_content_value(content: dict, key: str, default=""):
    """Handle OpenReview v1/v2 content wrapping difference defensively."""
    val = content.get(key, default)
    if isinstance(val, dict):
        return val.get("value", default)
    return val if val is not None else default


def fetch_openreview_venue(
    venue_id: str,
    venue_label: str,
    source: str,
    keywords: list[str],
    max_results: int = 0,
) -> list[dict]:
    """
    Fetch accepted papers from OpenReview, filter by keywords,
    and return in paper_radar raw dict format.

    Args:
        venue_id: OpenReview venue ID (e.g. "robot-learning.org/CoRL/2024/Conference")
        venue_label: display label (e.g. "CoRL 2024")
        source: source identifier ("corl" | "rss" | "neurips")
        keywords: robotics keyword list (for filtering)
        max_results: 0 means unlimited

    Returns:
        list of paper dicts compatible with Paper dataclass
    """
    try:
        import openreview
    except ImportError:
        print("[ERROR] openreview-py package is not installed. "
              "Run: pip install openreview-py")
        return []

    try:
        client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net"
        )
    except Exception as e:
        print(f"[ERROR] OpenReview client creation failed: {e}")
        return []

    # Query accepted papers
    try:
        notes = list(openreview.tools.iterget_notes(
            client, content={"venueid": venue_id}
        ))
    except Exception as e:
        print(f"[ERROR] OpenReview query failed (venue={venue_id}): {e}")
        return []

    print(f"  [OpenReview] {venue_label}: {len(notes)} accepted papers found")

    if max_results > 0:
        notes = notes[:max_results]

    # Lowercase keywords once for matching
    keywords_lower = [kw.lower() for kw in keywords]

    results = []
    for note in notes:
        content = note.content or {}
        title = _get_content_value(content, "title", "")
        abstract = _get_content_value(content, "abstract", "")

        if not title:
            continue

        # Keyword filter: at least one keyword in title + abstract
        text = (title + " " + abstract).lower()
        matched = [kw for kw, kw_l in zip(keywords, keywords_lower) if kw_l in text]
        if not matched:
            continue

        # Extract authors
        authors_raw = _get_content_value(content, "authors", [])
        if isinstance(authors_raw, list):
            authors = ", ".join(authors_raw)
        else:
            authors = str(authors_raw)

        # Try to extract arXiv ID (OpenReview note may contain arXiv link)
        arxiv_id = ""
        arxiv_url = ""
        # Check PDF URL for arXiv link
        pdf_val = _get_content_value(content, "pdf", "")
        if isinstance(pdf_val, str) and "arxiv.org" in pdf_val:
            m = re.search(r"(\d{4}\.\d{4,5})", pdf_val)
            if m:
                arxiv_id = m.group(1)
                arxiv_url = f"http://arxiv.org/abs/{arxiv_id}"

        # Determine paper_id
        note_id = note.id or ""
        if arxiv_id:
            paper_id = f"arxiv:{arxiv_id}"
        else:
            paper_id = f"openreview:{note_id}"

        # publish_date: use note's cdate (creation date), fallback to venue year
        publish_date = ""
        if hasattr(note, "cdate") and note.cdate:
            import datetime
            try:
                # cdate is a millisecond timestamp
                dt = datetime.datetime.fromtimestamp(note.cdate / 1000)
                publish_date = dt.strftime("%Y-%m-%d")
            except (ValueError, OSError):
                pass
        if not publish_date:
            # Extract year from venue_label (e.g. "CoRL 2024" -> "2024-01-01")
            year_match = re.search(r"(\d{4})", venue_label)
            publish_date = f"{year_match.group(1)}-01-01" if year_match else "2024-01-01"

        # OpenReview URL
        openreview_url = f"https://openreview.net/forum?id={note_id}" if note_id else ""
        project_url = openreview_url

        results.append({
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract.replace("\n", " "),
            "authors": authors,
            "publish_date": publish_date,
            "arxiv_url": arxiv_url,
            "project_url": project_url,
            "source": source,
            "venue": venue_label,
            "matched_keywords": matched,
        })

    return results


# ─────────────────────────────────────────────────────────────
# Shared helpers for HTML-based venues (CVF Open Access, RSS)
# ─────────────────────────────────────────────────────────────

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; paper-radar/1.0)",
}
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


def _http_get(url: str, timeout: int = 30, retries: int = 3) -> str:
    """GET a page and return its text ("" on failure after retries)."""
    import time
    import requests

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] GET failed ({url}): {e}")
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _match_keywords(title: str, abstract: str, keywords: list[str]) -> list[str]:
    text = (title + " " + abstract).lower()
    return [kw for kw in keywords if kw.lower() in text]


def _fallback_date(venue_label: str, month_day: str = "01-01") -> str:
    year_match = re.search(r"(\d{4})", venue_label)
    year = year_match.group(1) if year_match else "2024"
    return f"{year}-{month_day}"


def _fetch_many(urls: list[str], workers: int = 8) -> dict[str, str]:
    """Fetch pages concurrently; returns {url: html}."""
    from concurrent.futures import ThreadPoolExecutor

    results: dict[str, str] = {}
    if not urls:
        return results
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (url, html) in enumerate(zip(urls, pool.map(_http_get, urls)), 1):
            results[url] = html
            if i % 200 == 0 or i == len(urls):
                print(f"    → {i}/{len(urls)} pages fetched")
    return results


def _make_result(
    *,
    paper_id: str,
    title: str,
    abstract: str,
    authors: str,
    publish_date: str,
    project_url: str,
    arxiv_id: str,
    source: str,
    venue_label: str,
    matched: list[str],
) -> dict:
    return {
        "paper_id": f"arxiv:{arxiv_id}" if arxiv_id else paper_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publish_date": publish_date,
        "arxiv_url": f"http://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        "project_url": project_url,
        "source": source,
        "venue": venue_label,
        "matched_keywords": matched,
    }


# ─────────────────────────────────────────────────────────────
# CVF Open Access (CVPR / ICCV / WACV)
# ─────────────────────────────────────────────────────────────

def fetch_cvf_venue(
    url: str,
    venue_label: str,
    source: str,
    keywords: list[str],
    max_results: int = 0,
    workers: int = 8,
    publish_date: str = "",
) -> list[dict]:
    """
    Fetch accepted papers from CVF Open Access
    (e.g. https://openaccess.thecvf.com/CVPR2026?day=all).

    The listing page only has title/authors/links, so each paper's HTML page
    is fetched for the abstract (concurrently, `workers` threads).
    """
    from urllib.parse import urljoin
    from bs4 import BeautifulSoup

    html = _http_get(url, timeout=60)
    if not html:
        print(f"[ERROR] CVF listing not available: {url}")
        return []
    soup = BeautifulSoup(html, "html.parser")

    entries = []
    for dt in soup.select("dt.ptitle"):
        a = dt.find("a", href=True)
        if not a:
            continue
        title = _clean_text(a.get_text())
        page_url = urljoin(url, a["href"])

        # Following <dd> siblings: authors, then links (pdf/supp/arXiv/bibtex)
        authors, arxiv_id, pdf_url = [], "", ""
        for dd in dt.find_next_siblings():
            if dd.name != "dd":
                break
            for inp in dd.select('input[name="query_author"]'):
                if inp.get("value"):
                    authors.append(_clean_text(inp["value"]))
            for link in dd.find_all("a", href=True):
                href = link["href"]
                m = _ARXIV_ID_RE.search(href)
                if m:
                    arxiv_id = m.group(1)
                elif href.endswith("_paper.pdf"):
                    pdf_url = urljoin(url, href)
        if not authors:
            # Fallback: author names rendered as plain links in the first <dd>
            dd = dt.find_next_sibling("dd")
            if dd:
                authors = [
                    _clean_text(x.get_text()) for x in dd.find_all("a")
                    if _clean_text(x.get_text())
                ]
        entries.append({
            "title": title,
            "page_url": page_url,
            "authors": authors,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
        })

    print(f"  [CVF] {venue_label}: {len(entries)} accepted papers found")
    if not entries:
        print("  [WARN] No dt.ptitle entries; is the URL correct "
              "(e.g. ...?day=all) and the proceedings published?")
    if max_results > 0:
        entries = entries[:max_results]

    print(f"  [CVF] Fetching {len(entries)} paper pages for abstracts "
          f"({workers} workers)...")
    pages = _fetch_many([e["page_url"] for e in entries], workers=workers)

    date = publish_date or _fallback_date(venue_label, "06-01")
    results = []
    for e in entries:
        abstract = ""
        page_html = pages.get(e["page_url"], "")
        if page_html:
            page = BeautifulSoup(page_html, "html.parser")
            abs_div = page.find(id="abstract")
            if abs_div:
                abstract = _clean_text(abs_div.get_text())
            if not e["arxiv_id"]:
                for link in page.find_all("a", href=True):
                    m = _ARXIV_ID_RE.search(link["href"])
                    if m:
                        e["arxiv_id"] = m.group(1)
                        break

        matched = _match_keywords(e["title"], abstract, keywords)
        if not matched:
            continue

        slug = e["page_url"].rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
        results.append(_make_result(
            paper_id=f"cvf:{slug}",
            title=e["title"],
            abstract=abstract,
            authors=", ".join(e["authors"]),
            publish_date=date,
            project_url=e["page_url"],
            arxiv_id=e["arxiv_id"],
            source=source,
            venue_label=venue_label,
            matched=matched,
        ))
    return results


# ─────────────────────────────────────────────────────────────
# RSS proceedings (roboticsproceedings.org)
# ─────────────────────────────────────────────────────────────

def _parse_rss_paper_page(html: str) -> dict:
    """
    Best-effort parser for a roboticsproceedings.org paper page (pNNN.html).
    Layout: volume header, <h3>title</h3>, <i>authors</i>, an "Abstract"
    label followed by the abstract paragraph, and a PDF link.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    info = {"title": "", "authors": "", "abstract": "", "arxiv_id": ""}

    for tag in soup.find_all(["h3", "h2", "h1"]):
        text = _clean_text(tag.get_text())
        if text and "Robotics: Science and Systems" not in text:
            info["title"] = text
            authors_tag = tag.find_next("i")
            if authors_tag:
                info["authors"] = _clean_text(authors_tag.get_text())
            break

    label = soup.find(string=re.compile(r"^\s*Abstract\s*:?\s*$", re.I))
    if label is not None:
        # Abstract text is the next non-empty text block after the label
        node = label.parent
        for nxt in node.find_all_next(string=True):
            text = _clean_text(nxt)
            if len(text) > 80:
                info["abstract"] = _clean_text(nxt.parent.get_text())
                break
    if not info["abstract"]:
        paragraphs = [_clean_text(p.get_text()) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p) > 200]
        if paragraphs:
            info["abstract"] = max(paragraphs, key=len)
    info["abstract"] = re.sub(r"^Abstract\s*:?\s*", "", info["abstract"], flags=re.I)

    for link in soup.find_all("a", href=True):
        m = _ARXIV_ID_RE.search(link["href"])
        if m:
            info["arxiv_id"] = m.group(1)
            break
    return info


def fetch_rss_proceedings(
    url: str,
    venue_label: str,
    source: str,
    keywords: list[str],
    max_results: int = 0,
    workers: int = 8,
    publish_date: str = "",
) -> list[dict]:
    """
    Fetch accepted papers from the RSS proceedings site
    (e.g. https://www.roboticsproceedings.org/rss22/index.html for RSS 2026).
    """
    from urllib.parse import urljoin
    from bs4 import BeautifulSoup

    html = _http_get(url, timeout=60)
    if not html:
        print(f"[ERROR] RSS proceedings index not available: {url}")
        return []
    soup = BeautifulSoup(html, "html.parser")

    entries: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(r"(?:^|/)p\d{3}\.html$", href):
            continue
        page_url = urljoin(url, href)
        if page_url in entries:
            continue
        # Authors are usually the <i> right after the title link in the index
        authors = ""
        nxt = a.find_next(["i", "a"])
        if nxt is not None and nxt.name == "i":
            authors = _clean_text(nxt.get_text())
        entries[page_url] = {
            "page_url": page_url,
            "title": _clean_text(a.get_text()),
            "authors": authors,
        }

    entry_list = list(entries.values())
    print(f"  [RSS] {venue_label}: {len(entry_list)} accepted papers found")
    if max_results > 0:
        entry_list = entry_list[:max_results]

    print(f"  [RSS] Fetching {len(entry_list)} paper pages for abstracts...")
    pages = _fetch_many([e["page_url"] for e in entry_list], workers=workers)

    year_match = re.search(r"(\d{4})", venue_label)
    year = year_match.group(1) if year_match else ""
    date = publish_date or _fallback_date(venue_label, "07-01")
    results = []
    for e in entry_list:
        info = _parse_rss_paper_page(pages.get(e["page_url"], "")) \
            if pages.get(e["page_url"]) else {}
        title = info.get("title") or e["title"]
        authors = info.get("authors") or e["authors"]
        abstract = info.get("abstract", "")
        if not title:
            continue

        matched = _match_keywords(title, abstract, keywords)
        if not matched:
            continue

        num = re.search(r"p(\d{3})\.html$", e["page_url"]).group(1)
        results.append(_make_result(
            paper_id=f"rss:{year}-p{num}",
            title=title,
            abstract=abstract,
            authors=authors,
            publish_date=date,
            project_url=e["page_url"],
            arxiv_id=info.get("arxiv_id", ""),
            source=source,
            venue_label=venue_label,
            matched=matched,
        ))
    return results


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────

def _apply_require_any(results: list[dict], terms: list[str], label: str) -> list[dict]:
    """Keep papers whose title+abstract contains at least one of `terms`."""
    if not terms:
        return results
    terms_l = [t.lower() for t in terms]
    kept = [
        r for r in results
        if any(t in (r["title"] + " " + r["abstract"]).lower() for t in terms_l)
    ]
    print(f"  [{label}] require_any filter: {len(results)} → {len(kept)} papers")
    return kept


def fetch_venue(venue_cfg: dict, keywords: list[str], max_results: int = 0) -> list[dict]:
    """
    Dispatch on venue_cfg["fetcher"]:
      - "openreview" (default): needs venue_id
      - "cvf":                  needs url (CVF Open Access listing, ?day=all)
      - "rss":                  needs url (roboticsproceedings.org index)
    Optional venue_cfg["require_any"]: extra filter applied after the keyword
    match (title+abstract must also contain one of these terms).
    """
    results = _fetch_venue_raw(venue_cfg, keywords, max_results)
    return _apply_require_any(
        results, venue_cfg.get("require_any") or [], venue_cfg["label"]
    )


def _fetch_venue_raw(venue_cfg: dict, keywords: list[str], max_results: int = 0) -> list[dict]:
    fetcher = venue_cfg.get("fetcher", "openreview")
    common = dict(
        venue_label=venue_cfg["label"],
        source=venue_cfg["source"],
        keywords=keywords,
        max_results=max_results,
    )
    if fetcher == "openreview":
        return fetch_openreview_venue(venue_id=venue_cfg["venue_id"], **common)
    extra = dict(
        url=venue_cfg["url"],
        workers=int(venue_cfg.get("workers", 8)),
        publish_date=venue_cfg.get("publish_date", ""),
    )
    if fetcher == "cvf":
        return fetch_cvf_venue(**common, **extra)
    if fetcher == "rss":
        return fetch_rss_proceedings(**common, **extra)
    print(f"[ERROR] Unknown conference fetcher '{fetcher}' for {venue_cfg['label']}")
    return []
