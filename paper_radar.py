"""
paper_radar.py  ─  Improved robotics paper tracker
────────────────────────────────────────────────────
Improvements over cold-young/robotics_paper_daily:
  1. Cross-category deduplication by arXiv ID
  2. HuggingFace Daily Papers integration
  3. Relevance score-based sorting
  4. Papers With Code API integration (auto code link collection)
  5. Full config.yaml compatibility
  6. Cumulative DB (docs/papers_db.json)
     - New papers accumulate daily; oldest papers are pruned
       when per-category limit is exceeded (default: 50/category)
  7. Conference paper collection via OpenReview (CoRL, NeurIPS, etc.)

Usage:
    python paper_radar.py                  # collect today, accumulate into DB
    python paper_radar.py --days 3         # collect last 3 days
    python paper_radar.py --reset-db       # reset DB and collect fresh
    python paper_radar.py --output my.md   # specify output file
    python paper_radar.py --conferences    # include conference papers (OpenReview)
"""

import re
import time
import json
import random
import argparse
import datetime
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dataclasses
import yaml   # pip install pyyaml
import arxiv  # pip install arxiv

# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

def make_paper_id(source: str, raw_id: str) -> str:
    """Generate a universal paper_id: 'arxiv:2605.12345' or 'openreview:AbC123'"""
    return f"{source}:{raw_id}"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: str          # "First Author et al."
    publish_date: str     # "YYYY-MM-DD"
    arxiv_url: str
    project_url: str = ""
    # Papers With Code fields
    code_url: str = ""        # GitHub repo URL
    pwc_url: str = ""         # paperswithcode.com page
    framework: str = ""       # "PyTorch" / "TensorFlow" / ""
    # Collection metadata
    matched_keywords: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    hf_rank: Optional[int] = None   # HuggingFace daily rank (1-based)
    score: int = 0                   # relevance score (for sorting)
    # Conference/source extension fields
    paper_id: str = ""        # universal PK: "arxiv:2605.12345" | "openreview:AbC123"
    source: str = "arxiv"     # "arxiv" | "hf" | "corl" | "rss" | "neurips"
    venue: str = ""           # display label: "CoRL 2024", "" (arXiv)

    def compute_score(self):
        """Keyword hit count + HF rank weight + code availability bonus"""
        self.score = len(self.matched_keywords) * 10
        if self.hf_rank is not None:
            self.score += max(0, 30 - self.hf_rank)  # top-1=+29, top-30=+1
        if self.code_url:
            self.score += 5   # slight boost for papers with code

    def keyword_badges(self) -> str:
        badges = []
        if self.venue:
            badges.append(f"`📚 {self.venue}`")
        for cat in self.matched_categories:
            badges.append(f"`{cat}`")
        if self.hf_rank is not None:
            badges.append(f"🔥 HF#{self.hf_rank}")
        if self.code_url:
            badges.append("💻 Code")
        return " ".join(badges)

    def author_team(self) -> str:
        """Last Author Team format (preserves legacy output)"""
        parts = [a.strip() for a in self.authors.split(",")]
        if len(parts) >= 2:
            return f"{parts[-1]} Team"
        return parts[0]

# ─────────────────────────────────────────────
# Papers With Code fetcher
# ─────────────────────────────────────────────

# Uses config.yaml base_url as-is
PWC_BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"
ARXIV_API    = "https://export.arxiv.org/api/query"  # fallback
ARXIV_DEFAULT_DELAY_SECONDS = 5.0
ARXIV_DEFAULT_RETRIES = 4
ARXIV_DEFAULT_PAGE_SIZE = 20
ARXIV_DEFAULT_BACKOFF_SECONDS = (30.0, 60.0, 120.0, 240.0)
ARXIV_HF_BATCH_SIZE = 10
CONFERENCE_KEYWORD_CATEGORIES = {
    "dexterous",
    "manipulation",
    "learnedcontrol",
    "sim2real",
    "tactile",
    "vla",
}


def _sleep_with_jitter(seconds: float) -> None:
    """Avoid synchronized retries from CI runners that share outbound IPs."""
    time.sleep(seconds + random.uniform(0.0, min(3.0, seconds * 0.1)))


def _is_arxiv_429(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 429:
        return True
    return "HTTP 429" in str(exc)


def _arxiv_backoff_seconds(attempt: int, backoff_seconds: tuple[float, ...]) -> float:
    if not backoff_seconds:
        return ARXIV_DEFAULT_BACKOFF_SECONDS[-1]
    if attempt < len(backoff_seconds):
        return backoff_seconds[attempt]
    return backoff_seconds[-1]


def fetch_pwc_by_id(arxiv_id: str) -> dict:
    """
    Query Papers With Code API for a single paper's code info.
    Example response:
      {
        "paper": {"id": "2603.09761", "title": "...", ...},
        "repository": {"url": "https://github.com/...", "framework": "PyTorch"},
        "paper_with_code": {"url": "https://paperswithcode.com/paper/..."}
      }
    Returns empty dict on failure.
    """
    url = f"{PWC_BASE_URL}{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "PaperRadar/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def enrich_with_pwc(paper: "Paper") -> None:
    """
    Enrich a Paper object in-place with PWC code info.
    """
    data = fetch_pwc_by_id(paper.arxiv_id)
    if not data:
        return

    repo = data.get("repository") or {}
    if repo.get("url"):
        paper.code_url  = repo["url"]
        paper.framework = repo.get("framework", "")

    pwc = data.get("paper_with_code") or {}
    if pwc.get("url"):
        paper.pwc_url = pwc["url"]


# ─────────────────────────────────────────────
# arXiv fetcher (fallback / bulk search)
# ─────────────────────────────────────────────

def _build_query_string(keywords: list[str]) -> str:
    """
    Convert keyword list to arXiv search query string.
      - Multi-word keywords are quoted
      - Single-word keywords are used as-is
      - Joined with OR
    """
    ESCAPE = '"'
    parts = []
    for kw in keywords:
        if len(kw.split()) > 1:
            parts.append(ESCAPE + kw + ESCAPE)
        else:
            parts.append(kw)
    return " OR ".join(parts)


def fetch_arxiv(
    keywords: list[str],
    max_results: int = 20,
    days_back: int = 1,
    chunk_size: int = 3,
    delay_seconds: float = ARXIV_DEFAULT_DELAY_SECONDS,
    retry_attempts: int = ARXIV_DEFAULT_RETRIES,
    page_size: int = ARXIV_DEFAULT_PAGE_SIZE,
    backoff_seconds: tuple[float, ...] = ARXIV_DEFAULT_BACKOFF_SECONDS,
) -> list[dict]:
    """
    Search using the arxiv Python library.
    Fetches max_results papers sorted by submission date (no date filter).
    """
    query = _build_query_string(keywords)

    for attempt in range(retry_attempts + 1):
        search_engine = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        seen: set[str] = set()
        results: list[dict] = []

        try:
            # arxiv>=2.0: Search.results() was removed; use Client().results(search).
            # Keep backward compatibility with older versions that still expose .results().
            if hasattr(arxiv, "Client"):
                # Keep library retries at zero so 429 handling uses our longer backoff.
                client = arxiv.Client(
                    page_size=max(1, min(page_size, max_results)),
                    delay_seconds=delay_seconds,
                    num_retries=0,
                )
                result_iter = client.results(search_engine)
            else:
                result_iter = search_engine.results()

            for result in result_iter:
                paper_id = result.get_short_id()
                ver_pos = paper_id.find("v")
                arxiv_id = paper_id[:ver_pos] if ver_pos != -1 else paper_id

                if arxiv_id in seen:
                    continue
                seen.add(arxiv_id)

                paper_url = f"http://arxiv.org/abs/{arxiv_id}"
                published = result.published.date().isoformat()

                # Extract GitHub/project URL from comments
                repo_url = ""
                project_url = ""
                if result.comment:
                    urls = re.findall(r"(https?://[^\s,;]+)", result.comment)
                    for url in urls:
                        if "github.com" in url or "gitlab.com" in url:
                            repo_url = url
                        else:
                            project_url = url

                authors_str = ", ".join(str(a) for a in result.authors)

                results.append({
                    "arxiv_id":     arxiv_id,
                    "title":        result.title,
                    "abstract":     result.summary.replace("\n", " "),
                    "authors":      authors_str,
                    "publish_date": published,
                    "arxiv_url":    paper_url,
                    "project_url":  repo_url or project_url,
                })
            return results
        except Exception as e:
            if _is_arxiv_429(e) and attempt < retry_attempts:
                wait = _arxiv_backoff_seconds(attempt, backoff_seconds)
                print(
                    f"  [WARN] arXiv rate limited (HTTP 429). "
                    f"Retrying in {wait:.0f}s ({attempt + 1}/{retry_attempts})..."
                )
                _sleep_with_jitter(wait)
                continue

            print(f"  [WARN] arXiv search skipped ({type(e).__name__}): {e}")
            break

    if not results:
        print(f"  [WARN] arXiv returned 0 results (query: {query[:50]}...) "
              f"— possible rate limit, library compatibility, or network issue")

    return results


# ─────────────────────────────────────────────
# HuggingFace Daily Papers fetcher
# ─────────────────────────────────────────────

HF_DAILY_API = "https://huggingface.co/api/daily_papers"

def fetch_hf_daily(limit: int = 50) -> dict[str, int]:
    """
    HuggingFace Daily Papers API -> {arxiv_id: rank} dict.
    Called without date parameter (date param causes 400 error).
    """
    params = {"limit": limit}
    url = HF_DAILY_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "PaperRadar/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[WARN] HuggingFace API access failed: {e}")
        return {}

    hf_map = {}
    for rank, item in enumerate(data, start=1):
        paper = item.get("paper", {})
        pid   = paper.get("id", "")   # e.g. "2403.12345"
        if pid:
            hf_map[pid] = rank
    return hf_map



# ─────────────────────────────────────────────
# Cumulative DB (docs/papers_db.json)
# ─────────────────────────────────────────────

DB_DEFAULT_PATH = "docs/papers_db.json"


def _paper_to_dict(p: Paper) -> dict:
    return dataclasses.asdict(p)


def _paper_from_dict(d: dict) -> Paper:
    # Backward compat: old DB entries may lack paper_id/source/venue fields
    if "paper_id" not in d or not d.get("paper_id"):
        d["paper_id"] = make_paper_id("arxiv", d["arxiv_id"])
    if "source" not in d:
        d["source"] = "arxiv"
    if "venue" not in d:
        d["venue"] = ""
    return Paper(**d)


def load_db(db_path: str = DB_DEFAULT_PATH) -> dict[str, Paper]:
    """
    Load JSON DB -> {paper_id: Paper}.
    Backward compatible: old DB keyed by arxiv_id is migrated to paper_id keys.
    """
    path = Path(db_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result = {}
        for _key, d in raw.items():
            p = _paper_from_dict(d)
            result[p.paper_id] = p
        return result
    except Exception as e:
        print(f"[WARN] DB load failed ({db_path}): {e} — starting with empty DB")
        return {}


def save_db(papers: dict[str, Paper], db_path: str = DB_DEFAULT_PATH) -> None:
    """
    Save {paper_id: Paper} -> JSON DB.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {pid: _paper_to_dict(p) for pid, p in papers.items()}
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_into_db(
    db: dict[str, Paper],
    new_papers: dict[str, Paper],
    max_per_category: int = 50,
    max_hf_hot_only: int = 200,
) -> dict[str, Paper]:
    """
    Merge new papers into existing DB, then prune oldest papers
    when per-category count exceeds max_per_category.

    Rules:
      - If paper_id already exists, update category/keyword tags
        (also refresh code links, HF rank, etc.)
      - Otherwise, add as new entry
      - Keep only newest N papers per category
        (papers in multiple categories count toward each)
    """
    merged = dict(db)  # copy existing DB

    # 1. Merge new papers
    for pid, new_p in new_papers.items():
        if pid in merged:
            old_p = merged[pid]
            # Accumulate category/keyword tags
            for cat in new_p.matched_categories:
                if cat not in old_p.matched_categories:
                    old_p.matched_categories.append(cat)
            for kw in new_p.matched_keywords:
                if kw not in old_p.matched_keywords:
                    old_p.matched_keywords.append(kw)
            # Refresh with new info (code links, HF rank, etc.)
            if new_p.code_url:
                old_p.code_url  = new_p.code_url
                old_p.framework = new_p.framework
            if new_p.pwc_url:
                old_p.pwc_url = new_p.pwc_url
            if new_p.hf_rank is not None:
                old_p.hf_rank = new_p.hf_rank
        else:
            merged[pid] = new_p

    # 2. Enforce per-category max (prune oldest)
    #    Collect all categories (exclude HF-Hot: changes daily)
    all_cats: set[str] = set()
    for p in merged.values():
        for cat in p.matched_categories:
            if cat != "HF-Hot":
                all_cats.add(cat)

    papers_to_remove: set[str] = set()
    for cat in all_cats:
        cat_papers = [
            p for p in merged.values()
            if cat in p.matched_categories
        ]
        if len(cat_papers) <= max_per_category:
            continue
        # Sort by date ascending; overflow (oldest) are removal candidates
        cat_papers_sorted = sorted(cat_papers, key=lambda p: p.publish_date)
        overflow = len(cat_papers) - max_per_category
        for old_p in cat_papers_sorted[:overflow]:
            # Papers in other categories: only remove this category tag
            old_p.matched_categories = [
                c for c in old_p.matched_categories if c != cat
            ]
            # Papers with no remaining categories are fully removed
            if not old_p.matched_categories:
                papers_to_remove.add(old_p.paper_id)

    for pid in papers_to_remove:
        del merged[pid]

    # 3. Prune stale HF-only papers. These arrive daily and otherwise dominate
    #    the DB while not belonging to a long-lived robotics category.
    if max_hf_hot_only > 0:
        hf_only_papers = [
            p for p in merged.values()
            if set(p.matched_categories) == {"HF-Hot"}
        ]
        if len(hf_only_papers) > max_hf_hot_only:
            hf_only_sorted = sorted(
                hf_only_papers,
                key=lambda p: (p.publish_date, p.paper_id),
                reverse=True,
            )
            for old_p in hf_only_sorted[max_hf_hot_only:]:
                merged.pop(old_p.paper_id, None)

    return merged


def get_display_papers(
    db: dict[str, Paper],
    hf_map: dict[str, int],
) -> dict[str, Paper]:
    """
    Return display-ready paper dict from DB.
    HF rank is refreshed to today's values (cumulative HF rank is meaningless).
    """
    display = {}
    for pid, p in db.items():
        # Copy Paper (keep original DB immutable)
        dp = Paper(**dataclasses.asdict(p))
        # Overwrite HF rank with today's data (hf_map is keyed by arxiv_id)
        dp.hf_rank = hf_map.get(dp.arxiv_id, None)
        dp.compute_score()
        display[pid] = dp
    return display

# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load config.yaml (compatible with legacy format):
      categories:
        Dexterous:
          keywords: [dexterous, tactile, ...]
        Manipulation:
          keywords: [manipulation, grasping, ...]
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────
# Core: collect & merge
# ─────────────────────────────────────────────

def collect_papers(config: dict, include_conferences: bool = False, **kwargs) -> tuple[
    dict[str, Paper],   # all papers (deduped), keyed by paper_id
    dict[str, int],     # hf_map
]:
    categories: dict[str, list[str]] = {}
    for cat_name, cat_cfg in config.get("categories", {}).items():
        categories[cat_name] = cat_cfg.get("keywords", [])

    settings = config.get("settings", {})
    max_results_per_category = int(settings.get("max_results_per_category", 20))
    hf_daily_limit = int(settings.get("hf_daily_limit", 50))
    arxiv_delay_seconds = float(
        settings.get("arxiv_delay_seconds", ARXIV_DEFAULT_DELAY_SECONDS)
    )
    arxiv_retry_attempts = int(
        settings.get("arxiv_retry_attempts", ARXIV_DEFAULT_RETRIES)
    )
    arxiv_page_size = int(
        settings.get(
            "arxiv_page_size",
            min(ARXIV_DEFAULT_PAGE_SIZE, max_results_per_category),
        )
    )
    arxiv_hf_batch_size = int(settings.get("arxiv_hf_batch_size", ARXIV_HF_BATCH_SIZE))
    arxiv_page_size = max(1, arxiv_page_size)
    arxiv_hf_batch_size = max(1, arxiv_hf_batch_size)
    arxiv_backoff_seconds = tuple(
        float(v) for v in settings.get(
            "arxiv_backoff_seconds",
            ARXIV_DEFAULT_BACKOFF_SECONDS,
        )
    )

    # 1. HuggingFace hot papers
    hf_map = fetch_hf_daily(limit=hf_daily_limit)
    print(f"[HF] {len(hf_map)} daily papers fetched")

    # 2. arXiv per category
    all_papers: dict[str, Paper] = {}

    for cat_name, keywords in categories.items():
        print(f"[arXiv] Fetching category '{cat_name}' with {len(keywords)} keywords...")
        _sleep_with_jitter(arxiv_delay_seconds)

        raw = fetch_arxiv(
            keywords,
            max_results=max_results_per_category,
            delay_seconds=arxiv_delay_seconds,
            retry_attempts=arxiv_retry_attempts,
            page_size=arxiv_page_size,
            backoff_seconds=arxiv_backoff_seconds,
        )
        print(f"  → {len(raw)} papers before dedup")

        for r in raw:
            aid = r["arxiv_id"]
            pid = make_paper_id("arxiv", aid)

            # Check which keywords matched
            text = (r["title"] + " " + r["abstract"]).lower()
            matched = [kw for kw in keywords if kw.lower() in text]

            if pid not in all_papers:
                all_papers[pid] = Paper(
                    arxiv_id=aid,
                    title=r["title"],
                    abstract=r["abstract"],
                    authors=r["authors"],
                    publish_date=r["publish_date"],
                    arxiv_url=r["arxiv_url"],
                    project_url=r["project_url"],
                    paper_id=pid,
                    source="arxiv",
                )

            p = all_papers[pid]
            # Add new categories/keywords (no duplicates)
            if cat_name not in p.matched_categories:
                p.matched_categories.append(cat_name)
            for kw in matched:
                if kw not in p.matched_keywords:
                    p.matched_keywords.append(kw)

    # 3. Assign HF rank (hf_map is keyed by arxiv_id)
    for pid, paper in all_papers.items():
        if paper.arxiv_id in hf_map:
            paper.hf_rank = hf_map[paper.arxiv_id]

    # 4. Add HF hot papers not yet in collection (batched arXiv id lookup)
    # Convert hf_map keys (arxiv_id) to paper_id for dedup check
    existing_arxiv_ids = {p.arxiv_id for p in all_papers.values()}
    missing_hf_ids = [hf_id for hf_id in hf_map if hf_id not in existing_arxiv_ids]
    if missing_hf_ids:
        print(f"[arXiv] Fetching {len(missing_hf_ids)} missing HF papers by id...")
    for start in range(0, len(missing_hf_ids), arxiv_hf_batch_size):
        batch_ids = missing_hf_ids[start:start + arxiv_hf_batch_size]
        raws = fetch_arxiv_by_ids(
            batch_ids,
            delay_seconds=arxiv_delay_seconds,
            retry_attempts=arxiv_retry_attempts,
            backoff_seconds=arxiv_backoff_seconds,
        )
        for hf_id in batch_ids:
            raw = raws.get(_normalize_arxiv_id(hf_id))
            if not raw:
                continue
            rank = hf_map[hf_id]
            pid = make_paper_id("arxiv", raw["arxiv_id"])
            p = Paper(
                arxiv_id=raw["arxiv_id"],
                title=raw["title"],
                abstract=raw["abstract"],
                authors=raw["authors"],
                publish_date=raw["publish_date"],
                arxiv_url=raw["arxiv_url"],
                project_url=raw["project_url"],
                hf_rank=rank,
                matched_categories=["HF-Hot"],
                paper_id=pid,
                source="arxiv",
            )
            all_papers[pid] = p

    # 5. Conference paper collection (--conferences mode)
    if include_conferences:
        from conference_fetch import fetch_openreview_venue
        all_keywords = []
        for kws in categories.values():
            all_keywords.extend(kws)
        all_keywords = list(set(all_keywords))

        conf_cfg = config.get("conferences", {})
        if conf_cfg.get("enabled", False):
            max_conf = conf_cfg.get("max_results_per_venue", 0)
            for venue_cfg in conf_cfg.get("venues", []):
                source = venue_cfg["source"]
                label = venue_cfg["label"]
                venue_id = venue_cfg["venue_id"]
                print(f"\n[Conference] Fetching {label} (venue={venue_id})...")
                conf_papers = fetch_openreview_venue(
                    venue_id=venue_id,
                    venue_label=label,
                    source=source,
                    keywords=all_keywords,
                    max_results=max_conf,
                )
                print(f"  → {len(conf_papers)} papers after keyword filter")
                for r in conf_papers:
                    pid = r["paper_id"]
                    if pid in all_papers:
                        continue
                    # Assign categories by keyword matching
                    text = (r["title"] + " " + r["abstract"]).lower()
                    matched_cats = []
                    matched_kws = []
                    for cat_name, kws in categories.items():
                        cat_matched = [kw for kw in kws if kw.lower() in text]
                        if cat_matched:
                            matched_cats.append(cat_name)
                            matched_kws.extend(cat_matched)
                    if not matched_cats:
                        matched_cats = [label]  # fallback: venue as category
                    all_papers[pid] = Paper(
                        arxiv_id=r.get("arxiv_id", ""),
                        title=r["title"],
                        abstract=r["abstract"],
                        authors=r["authors"],
                        publish_date=r["publish_date"],
                        arxiv_url=r.get("arxiv_url", ""),
                        project_url=r.get("project_url", ""),
                        matched_categories=matched_cats,
                        matched_keywords=list(set(matched_kws)),
                        paper_id=pid,
                        source=source,
                        venue=label,
                    )

    # 6. Enrich with Papers With Code links
    #    0.5s delay to avoid rate limiting
    #    Skip conference papers without arXiv ID
    print(f"\n[PWC] Enriching {len(all_papers)} papers with code links...")
    pwc_count = 0
    for i, (pid, paper) in enumerate(all_papers.items()):
        if not paper.arxiv_id:
            continue  # skip conference papers without arXiv ID
        time.sleep(0.5)
        enrich_with_pwc(paper)
        if paper.code_url:
            pwc_count += 1
        if (i + 1) % 20 == 0:
            print(f"  → {i+1}/{len(all_papers)} processed, {pwc_count} with code")
    print(f"[PWC] Done. {pwc_count}/{len(all_papers)} papers have code links.")

    # 7. Final score computation (includes code bonus)
    for p in all_papers.values():
        p.compute_score()

    return all_papers, hf_map


def _normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.rsplit("/", 1)[-1])


def _entry_to_arxiv_dict(entry: ET.Element, ns: dict[str, str], arxiv_id: str) -> dict:
    title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
    abstract = entry.find("atom:summary", ns).text.replace("\n", " ").strip()
    published = entry.find("atom:published", ns).text[:10]
    authors = [a.find("atom:name", ns).text
               for a in entry.findall("atom:author", ns)]

    project_url = ""
    for link in entry.findall("atom:link", ns):
        href = link.get("href", "")
        if href.startswith("http") and "arxiv" not in href:
            project_url = href
            break

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": ", ".join(authors),
        "publish_date": published,
        "arxiv_url": f"http://arxiv.org/abs/{arxiv_id}",
        "project_url": project_url,
    }


def fetch_arxiv_by_ids(
    arxiv_ids: list[str],
    delay_seconds: float = ARXIV_DEFAULT_DELAY_SECONDS,
    retry_attempts: int = ARXIV_DEFAULT_RETRIES,
    backoff_seconds: tuple[float, ...] = ARXIV_DEFAULT_BACKOFF_SECONDS,
) -> dict[str, dict]:
    """Fetch multiple papers by arXiv ID using a single id_list request."""
    ids = [_normalize_arxiv_id(aid) for aid in arxiv_ids if aid]
    if not ids:
        return {}

    params = {"id_list": ",".join(ids)}
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "PaperRadar/1.0"})
    _sleep_with_jitter(delay_seconds)

    for attempt in range(retry_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                xml_data = resp.read()
            break
        except Exception as e:
            if _is_arxiv_429(e) and attempt < retry_attempts:
                wait = _arxiv_backoff_seconds(attempt, backoff_seconds)
                print(
                    f"  [WARN] arXiv id lookup rate limited (HTTP 429). "
                    f"Retrying in {wait:.0f}s ({attempt + 1}/{retry_attempts})..."
                )
                _sleep_with_jitter(wait)
                continue
            print(f"  [WARN] arXiv id lookup skipped ({type(e).__name__}): {e}")
            return {}

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    results: dict[str, dict] = {}
    for entry in root.findall("atom:entry", ns):
        entry_id = entry.find("atom:id", ns)
        if entry_id is None or not entry_id.text:
            continue
        arxiv_id = _normalize_arxiv_id(entry_id.text)
        results[arxiv_id] = _entry_to_arxiv_dict(entry, ns, arxiv_id)

    return results


def fetch_arxiv_by_id(arxiv_id: str) -> Optional[dict]:
    """Fetch a single paper by arXiv ID."""
    return fetch_arxiv_by_ids([arxiv_id]).get(_normalize_arxiv_id(arxiv_id))


# ─────────────────────────────────────────────
# Markdown generator
# ─────────────────────────────────────────────

def _abstract_short(abstract: str, max_len: int = 400) -> str:
    if len(abstract) <= max_len:
        return abstract
    return abstract[:max_len].rsplit(" ", 1)[0] + "..."


def _paper_links(p: Paper) -> str:
    """Build markdown links for arXiv/OpenReview/code/PWC."""
    if p.arxiv_url:
        links = f"[ArXiv]({p.arxiv_url})"
    elif p.project_url:
        links = f"[OpenReview]({p.project_url})"
    else:
        links = ""
    if p.code_url:
        links += f" / [Code]({p.code_url})"
    elif p.project_url and p.arxiv_url:
        links += f" / [Web]({p.project_url})"
    if p.pwc_url:
        links += f" / [PWC]({p.pwc_url})"
    return links


def _conference_labels(all_papers: dict[str, Paper], config: dict) -> list[str]:
    labels: list[str] = []
    for venue_cfg in config.get("conferences", {}).get("venues", []):
        label = venue_cfg.get("label", "")
        if label and label not in labels:
            labels.append(label)
    for p in all_papers.values():
        if p.venue and p.venue not in labels:
            labels.append(p.venue)
    return labels


def _display_limit(config: dict, key: str, default: int) -> int:
    return int(config.get("display", {}).get(key, default))


def _limit_papers(papers: list[Paper], limit: int) -> list[Paper]:
    return papers if limit <= 0 else papers[:limit]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _gitpage_nav_items(all_papers: dict[str, Paper], config: dict) -> list[tuple[str, str]]:
    items = [("HF Hot", "{{ site.baseurl }}/")]
    for label in _conference_labels(all_papers, config):
        if any(p.venue == label for p in all_papers.values()):
            items.append((
                label,
                f"{{{{ site.baseurl }}}}/conferences/{_slugify(label)}.html",
            ))
    for cat_name in config.get("categories", {}).keys():
        items.append((cat_name, f"{{{{ site.baseurl }}}}/{_slugify(cat_name)}.html"))
    return items


def _gitpage_nav(all_papers: dict[str, Paper], config: dict, active: str) -> str:
    styles = (
        "<style>"
        ".paper-nav{display:flex;flex-wrap:wrap;gap:.45rem;margin:1rem 0 1.25rem 0}"
        ".paper-nav a{border:1px solid #d0d7de;border-radius:999px;padding:.32rem .7rem;"
        "text-decoration:none;color:#24292f;background:#fff;font-size:.92rem}"
        ".paper-nav a.active{background:#0969da;color:#fff;border-color:#0969da}"
        "</style>"
    )
    links = []
    for label, href in _gitpage_nav_items(all_papers, config):
        cls = ' class="active"' if label == active else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    return styles + "\n\n" + '<nav class="paper-nav">' + "\n".join(links) + "</nav>"


def _gitpage_front_matter() -> list[str]:
    return ["---", "layout: default", "---", ""]


def _keyword_cell(p: Paper, max_keywords: int = 2) -> str:
    categories = [
        _slugify(cat)
        for cat in p.matched_categories
        if cat not in {"HF-Hot", p.venue}
    ]
    keywords = []
    for cat in categories:
        if cat in CONFERENCE_KEYWORD_CATEGORIES and cat not in keywords:
            keywords.append(cat)
    if not keywords:
        return ""
    return " ".join(f"`{kw}`" for kw in keywords[:max_keywords])


def _paper_row(p: Paper) -> str:
    """README table row format with PWC code links."""
    links = _paper_links(p)
    badges = p.keyword_badges()
    badge_str = f" {badges}" if badges else ""
    abstract_short = _abstract_short(p.abstract)

    return (
        f"| **{p.publish_date}** | "
        f"**{p.title}**{badge_str} "
        f"<details><summary>Abstract</summary>{abstract_short}</details> | "
        f"{p.author_team()} | "
        f"{links} |\n"
    )


def _conference_paper_row(p: Paper, details: bool = True) -> str:
    """Conference table row with matched keywords in the first column."""
    links = _paper_links(p)
    abstract_short = _abstract_short(p.abstract)
    if details:
        title_abstract = (
            f"**{p.title}** "
            f"<details><summary>Abstract</summary>{abstract_short}</details>"
        )
    else:
        title_abstract = f"**{p.title}**<br>{abstract_short}"
    return (
        f"| {_keyword_cell(p)} | "
        f"{title_abstract} | "
        f"{p.author_team()} | "
        f"{links} |"
    )


def generate_markdown(
    all_papers: dict[str, Paper],
    hf_map: dict[str, int],
    config: dict,
) -> str:
    today = datetime.date.today().strftime("%Y.%m.%d")
    cat_names = list(config.get("categories", {}).keys())
    conference_labels = _conference_labels(all_papers, config)
    max_hf = _display_limit(config, "readme_max_hf_hot", 30)
    max_cat = _display_limit(config, "readme_max_per_category", 40)
    max_conf = _display_limit(config, "readme_max_conference_per_venue", 50)

    lines = []
    lines.append(f"## Updated on {today}\n")

    # ── Table of Contents
    lines.append("## Table of Contents\n")
    lines.append("1. [🔥 HuggingFace Hot Papers](#-huggingface-hot-papers)")
    toc_items = cat_names + [
        label for label in conference_labels
        if any(p.venue == label for p in all_papers.values())
    ]
    for i, title in enumerate(toc_items, start=2):
        anchor = title.lower().replace(" ", "-")
        lines.append(f"{i}. [{title}](#{anchor})")
    lines.append("")

    # ── Section 1: HF Hot Papers
    lines.append("## 🔥 HuggingFace Hot Papers\n")
    hf_papers = _limit_papers(
        sorted(
            [p for p in all_papers.values() if p.hf_rank is not None],
            key=lambda p: p.hf_rank,
        ),
        max_hf,
    )

    if hf_papers:
        lines.append("<details><summary><b>HF Hot Papers (Click to expand)</b></summary>\n")
        lines.append("| Rank | Date | Title | Authors | Links |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in hf_papers:
            links = _paper_links(p)
            cat_tags = " ".join(f"`{c}`" for c in p.matched_categories if c != "HF-Hot")
            title_str = p.title + (f" {cat_tags}" if cat_tags else "")
            abstract_short = _abstract_short(p.abstract)
            lines.append(
                f"| 🔥{p.hf_rank} | **{p.publish_date}** | "
                f"**{title_str}** "
                f"<details><summary>Abstract</summary>{abstract_short}</details> | "
                f"{p.author_team()} | {links} |"
            )
        lines.append("\n</details>\n")
    else:
        lines.append("*No HuggingFace hot papers today or no robotics-related entries.*\n")

    # ── Section 2+: Per-category
    for cat_name in cat_names:
        anchor = cat_name.lower().replace(" ", "-")
        lines.append(f"## {cat_name}\n")

        cat_papers = _limit_papers(
            sorted(
                [
                    p for p in all_papers.values()
                    if cat_name in p.matched_categories and not p.venue
                ],
                key=lambda p: p.publish_date,
                reverse=True,
            ),
            max_cat,
        )

        lines.append(f"<details><summary><b>{cat_name} Papers (Click to expand)</b></summary>\n")
        lines.append("| Publish Date | Title & Abstract | Authors | Links |")
        lines.append("| --- | --- | --- | --- |")
        for p in cat_papers:
            lines.append(_paper_row(p).strip())
        lines.append("\n</details>\n")

    # ── Conference sections: separate from keyword categories
    for label in conference_labels:
        venue_papers = _limit_papers(
            sorted(
                [p for p in all_papers.values() if p.venue == label],
                key=lambda p: p.publish_date,
                reverse=True,
            ),
            max_conf,
        )
        if not venue_papers:
            continue
        lines.append(f"## {label}\n")
        lines.append(f"<details><summary><b>{label} Papers (Click to expand)</b></summary>\n")
        lines.append("| Keyword | Title & Abstract | Authors | Links |")
        lines.append("| --- | --- | --- | --- |")
        for p in venue_papers:
            lines.append(_conference_paper_row(p, details=True))
        lines.append("\n</details>\n")

    return "\n".join(lines)



# ─────────────────────────────────────────────
# GitPage Markdown generator (docs/index.md)
# Restored from daily_arxiv.py to_web=True logic
# ─────────────────────────────────────────────

def _paper_row_web(p: Paper) -> str:
    """GitHub Pages row — no <details>, includes back-to-top link"""
    links = _paper_links(p)
    badges = p.keyword_badges()
    badge_str = f" {badges}" if badges else ""
    abstract_short = _abstract_short(p.abstract)

    return (
        f"| **{p.publish_date}** | "
        f"**{p.title}**{badge_str}<br>{abstract_short} | "
        f"{p.author_team()} | "
        f"{links} |"
    )


def generate_gitpage_markdown(
    all_papers: dict[str, Paper],
    hf_map: dict[str, int],
    config: dict,
) -> str:
    """
    Generate GitHub Pages-compatible markdown (docs/index.md).
    Restored from daily_arxiv.py to_web=True format:
      - Jekyll front matter (layout: default)
      - No <details> — all content expanded
      - Back-to-top link at bottom of each section
      - Badges (contributors / forks / stars / issues)
    """
    max_hf = _display_limit(config, "gitpage_max_hf_hot", 30)
    hf_papers = _limit_papers(
        sorted(
            [p for p in all_papers.values() if p.hf_rank is not None],
            key=lambda p: p.hf_rank,
        ),
        max_hf,
    )
    return _generate_gitpage_section_page(
        all_papers,
        config,
        active="HF Hot",
        title="🔥 HuggingFace Hot Papers",
        papers=hf_papers,
        row_fn=_paper_row_web,
        table_header="| Publish Date | Title & Abstract | Authors | Links |",
        table_align="|:---------|:-----------------------|:---------|:------|",
        include_badges=True,
    )


def _badge_lines(config: dict) -> list[str]:
    user = config.get("user_name", "cold-young").replace(" ", "-")
    repo = config.get("repo_name", "robotics-paper-daily")
    return [
        "[![Contributors][contributors-shield]][contributors-url]",
        "[![Forks][forks-shield]][forks-url]",
        "[![Stargazers][stars-shield]][stars-url]",
        "[![Issues][issues-shield]][issues-url]",
        "",
        f"[contributors-shield]: https://img.shields.io/github/contributors/{user}/{repo}.svg?style=for-the-badge",
        f"[contributors-url]: https://github.com/{user}/{repo}/graphs/contributors",
        f"[forks-shield]: https://img.shields.io/github/forks/{user}/{repo}.svg?style=for-the-badge",
        f"[forks-url]: https://github.com/{user}/{repo}/network/members",
        f"[stars-shield]: https://img.shields.io/github/stars/{user}/{repo}.svg?style=for-the-badge",
        f"[stars-url]: https://github.com/{user}/{repo}/stargazers",
        f"[issues-shield]: https://img.shields.io/github/issues/{user}/{repo}.svg?style=for-the-badge",
        f"[issues-url]: https://github.com/{user}/{repo}/issues",
    ]


def _generate_gitpage_section_page(
    all_papers: dict[str, Paper],
    config: dict,
    active: str,
    title: str,
    papers: list[Paper],
    row_fn,
    table_header: str,
    table_align: str,
    include_badges: bool = False,
) -> str:
    today_dot = datetime.date.today().strftime("%Y.%m.%d")
    lines = _gitpage_front_matter()
    if include_badges:
        badge_lines = _badge_lines(config)
        lines.extend(badge_lines[:4])
        lines.append("")
    lines.append(f"## Updated on {today_dot}")
    lines.append(f"> Usage instructions: [here]({{{{ site.baseurl }}}}/README.html#usage)")
    lines.append("")
    lines.append(_gitpage_nav(all_papers, config, active))
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")
    lines.append(table_header)
    lines.append(table_align)
    for p in papers:
        lines.append(row_fn(p))
    if not papers:
        lines.append("|  | No papers available yet. |  |  |")
    if include_badges:
        lines.append("")
        lines.extend(_badge_lines(config)[5:])
    return "\n".join(lines)


def generate_gitpage_category_markdown(
    all_papers: dict[str, Paper],
    config: dict,
    cat_name: str,
) -> str:
    max_cat = _display_limit(config, "gitpage_max_per_category", 100)
    cat_papers = _limit_papers(
        sorted(
            [
                p for p in all_papers.values()
                if cat_name in p.matched_categories and not p.venue
            ],
            key=lambda p: p.publish_date,
            reverse=True,
        ),
        max_cat,
    )
    return _generate_gitpage_section_page(
        all_papers,
        config,
        active=cat_name,
        title=cat_name,
        papers=cat_papers,
        row_fn=_paper_row_web,
        table_header="| Publish Date | Title & Abstract | Authors | Links |",
        table_align="|:---------|:-----------------------|:---------|:------|",
    )


def generate_gitpage_conference_markdown(
    all_papers: dict[str, Paper],
    config: dict,
    venue_label: str,
) -> str:
    max_conf = _display_limit(config, "gitpage_max_conference_per_venue", 100)
    venue_papers = _limit_papers(
        sorted(
            [p for p in all_papers.values() if p.venue == venue_label],
            key=lambda p: p.publish_date,
            reverse=True,
        ),
        max_conf,
    )
    return _generate_gitpage_section_page(
        all_papers,
        config,
        active=venue_label,
        title=venue_label,
        papers=venue_papers,
        row_fn=lambda p: _conference_paper_row(p, details=False),
        table_header="| Keyword | Title & Abstract | Authors | Links |",
        table_align="|:---------|:-----------------------|:---------|:------|",
    )


def write_gitpage_markdowns(
    all_papers: dict[str, Paper],
    hf_map: dict[str, int],
    config: dict,
    gitpage_path: Path,
) -> list[Path]:
    written: list[Path] = []
    gitpage_path.parent.mkdir(parents=True, exist_ok=True)
    gitpage_path.write_text(
        generate_gitpage_markdown(all_papers, hf_map, config),
        encoding="utf-8",
    )
    written.append(gitpage_path)

    for cat_name in config.get("categories", {}).keys():
        path = gitpage_path.parent / f"{_slugify(cat_name)}.md"
        path.write_text(
            generate_gitpage_category_markdown(all_papers, config, cat_name),
            encoding="utf-8",
        )
        written.append(path)

    conf_dir = gitpage_path.parent / "conferences"
    conf_dir.mkdir(parents=True, exist_ok=True)
    for label in _conference_labels(all_papers, config):
        if not any(p.venue == label for p in all_papers.values()):
            continue
        path = conf_dir / f"{_slugify(label)}.md"
        path.write_text(
            generate_gitpage_conference_markdown(all_papers, config, label),
            encoding="utf-8",
        )
        written.append(path)

    return written


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaperRadar: robotics paper tracker")
    parser.add_argument("--config",     default="config.yaml",       help="config YAML path")
    parser.add_argument("--output",     default="README.md",         help="README output path")
    parser.add_argument("--gitpage",    default="docs/index.md",     help="GitPage output path")
    parser.add_argument("--db",         default="docs/papers_db.json", help="cumulative DB JSON path")
    parser.add_argument("--days",       type=int, default=1,         help="days back to collect")
    parser.add_argument("--max-per-cat",type=int, default=50,        help="max papers per category")
    parser.add_argument("--no-gitpage", action="store_true",         help="skip GitPage generation")
    parser.add_argument("--reset-db",   action="store_true",         help="reset DB and collect fresh")
    parser.add_argument("--conferences",action="store_true",         help="include conference papers (OpenReview)")
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Step 1: Collect new papers
    print("\n[Step 1] Collecting new papers...")
    today_papers, hf_map = collect_papers(
        config, include_conferences=args.conferences, days_back=args.days,
    )

    # ── Step 2: Load DB (or reset)
    if args.reset_db:
        print("[Step 2] DB reset (--reset-db flag)")
        db = {}
    else:
        print(f"[Step 2] Loading DB: {args.db}")
        db = load_db(args.db)
        print(f"         Existing DB: {len(db)} papers")

    # ── Step 3: Merge new papers into DB + prune overflow
    max_hf_hot_only = int(config.get("retention", {}).get("hf_hot_only_max", 200))
    print(f"[Step 3] Merging... (max {args.max_per_cat} per category)")
    db = merge_into_db(
        db,
        today_papers,
        max_per_category=args.max_per_cat,
        max_hf_hot_only=max_hf_hot_only,
    )
    print(f"         After merge: {len(db)} papers")

    # ── Step 4: Save DB
    save_db(db, args.db)
    print(f"[Step 4] DB saved: {args.db}")

    # ── Step 5: Prepare display papers (refresh HF rank to today)
    display_papers = get_display_papers(db, hf_map)

    # ── Statistics
    total      = len(display_papers)
    hf_count   = sum(1 for p in display_papers.values() if p.hf_rank is not None)
    multi_cat  = sum(1 for p in display_papers.values() if len(p.matched_categories) > 1)
    code_count = sum(1 for p in display_papers.values() if p.code_url)
    cat_stats  = {}
    for p in display_papers.values():
        for cat in p.matched_categories:
            if cat != "HF-Hot":
                cat_stats[cat] = cat_stats.get(cat, 0) + 1

    print(f"\n[Stats] Total papers: {total}")
    print(f"[Stats] HF hot papers     : {hf_count}")
    print(f"[Stats] Multi-category    : {multi_cat}")
    print(f"[Stats] With code links   : {code_count}")
    for cat, cnt in sorted(cat_stats.items()):
        bar = "█" * int(cnt / args.max_per_cat * 20)
        print(f"[Stats]   {cat:<18} {cnt:>3}/{args.max_per_cat}  {bar}")

    # ── Generate README.md
    readme_md = generate_markdown(display_papers, hf_map, config)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(readme_md, encoding="utf-8")
    print(f"\n✅ README  → {out_path}")

    # ── Generate docs/index.md (GitPage)
    if not args.no_gitpage:
        gitpage_path = Path(args.gitpage)
        written_pages = write_gitpage_markdowns(display_papers, hf_map, config, gitpage_path)
        print(f"✅ GitPage → {gitpage_path} (+{len(written_pages) - 1} section pages)")


if __name__ == "__main__":
    main()
