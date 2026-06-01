"""
conference_fetch.py  ─  OpenReview conference paper collector
─────────────────────────────────────────────────────────────
Collects accepted papers from conferences hosted on OpenReview
(CoRL, NeurIPS, etc.) and filters them by keywords.

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
