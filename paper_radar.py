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
import argparse
import datetime
import urllib.request
import urllib.parse
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
    return "OR".join(parts)


def fetch_arxiv(keywords: list[str], max_results: int = 20,
                days_back: int = 1, chunk_size: int = 3) -> list[dict]:
    """
    Search using the arxiv Python library.
    Fetches max_results papers sorted by submission date (no date filter).
    """
    query = _build_query_string(keywords)
    search_engine = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    # arxiv>=2.0: Search.results() was removed; use Client().results(search).
    # Keep backward compatibility with older versions that still expose .results().
    if hasattr(arxiv, "Client"):
        client = arxiv.Client(num_retries=3, delay_seconds=3.0)
        result_iter = client.results(search_engine)
    else:
        result_iter = search_engine.results()

    seen: set[str] = set()
    results: list[dict] = []

    try:
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
    except Exception as e:
        import traceback
        print(f"  [ERROR] arXiv search failed ({type(e).__name__}): {e}")
        traceback.print_exc()

    if not results:
        print(f"  [WARN] arXiv returned 0 results (query: {query[:50]}...) "
              f"— possible library compatibility or network issue")

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

    # 1. HuggingFace hot papers
    hf_map = fetch_hf_daily()
    print(f"[HF] {len(hf_map)} daily papers fetched")

    # 2. arXiv per category
    all_papers: dict[str, Paper] = {}

    for cat_name, keywords in categories.items():
        print(f"[arXiv] Fetching category '{cat_name}' with {len(keywords)} keywords...")
        time.sleep(3)  # respect arXiv API rate limit

        raw = fetch_arxiv(keywords, max_results=20)
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

    # 4. Add HF hot papers not yet in collection (fetch individually from arXiv)
    # Convert hf_map keys (arxiv_id) to paper_id for dedup check
    existing_arxiv_ids = {p.arxiv_id for p in all_papers.values()}
    for hf_id, rank in hf_map.items():
        if hf_id not in existing_arxiv_ids:
            try:
                time.sleep(1)
                raw = fetch_arxiv_by_id(hf_id)
                if raw:
                    pid = make_paper_id("arxiv", hf_id)
                    p = Paper(
                        arxiv_id=hf_id,
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
            except Exception as e:
                print(f"[WARN] HF paper {hf_id} fetch failed: {e}")

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


def fetch_arxiv_by_id(arxiv_id: str) -> Optional[dict]:
    """Fetch a single paper by arXiv ID"""
    url = f"{ARXIV_API}?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "PaperRadar/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_data = resp.read()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", ns)
    if not entries:
        return None

    entry = entries[0]
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


# ─────────────────────────────────────────────
# Markdown generator
# ─────────────────────────────────────────────

def _abstract_short(abstract: str, max_len: int = 400) -> str:
    if len(abstract) <= max_len:
        return abstract
    return abstract[:max_len].rsplit(" ", 1)[0] + "..."


def _paper_row(p: Paper) -> str:
    """README table row format with PWC code links."""
    # Links: arXiv URL if available, otherwise OpenReview/project URL
    if p.arxiv_url:
        links = f"[ArXiv]({p.arxiv_url})"
    elif p.project_url:
        links = f"[OpenReview]({p.project_url})"
    else:
        links = ""
    if p.code_url:
        links += f" / [Code]({p.code_url})"      # GitHub link preferred
    elif p.project_url and p.arxiv_url:
        links += f" / [Web]({p.project_url})"    # fallback to project page
    if p.pwc_url:
        links += f" / [PWC]({p.pwc_url})"        # Papers With Code page

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


def generate_markdown(
    all_papers: dict[str, Paper],
    hf_map: dict[str, int],
    config: dict,
) -> str:
    today = datetime.date.today().strftime("%Y.%m.%d")
    cat_names = list(config.get("categories", {}).keys())

    lines = []
    lines.append(f"## Updated on {today}\n")

    # ── Table of Contents
    lines.append("## Table of Contents\n")
    lines.append("1. [🔥 HuggingFace Hot Papers](#-huggingface-hot-papers)")
    for i, cat in enumerate(cat_names, start=2):
        anchor = cat.lower().replace(" ", "-")
        lines.append(f"{i}. [{cat}](#{anchor})")
    lines.append("")

    # ── Section 1: HF Hot Papers
    lines.append("## 🔥 HuggingFace Hot Papers\n")
    hf_papers = sorted(
        [p for p in all_papers.values() if p.hf_rank is not None],
        key=lambda p: p.hf_rank,
    )

    if hf_papers:
        lines.append("<details><summary><b>HF Hot Papers (Click to expand)</b></summary>\n")
        lines.append("| Rank | Date | Title | Authors | Links |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in hf_papers:
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

        cat_papers = sorted(
            [p for p in all_papers.values() if cat_name in p.matched_categories],
            key=lambda p: p.publish_date,
            reverse=True,
        )

        lines.append(f"<details><summary><b>{cat_name} Papers (Click to expand)</b></summary>\n")
        lines.append("| Publish Date | Title & Abstract | Authors | Links |")
        lines.append("| --- | --- | --- | --- |")
        for p in cat_papers:
            lines.append(_paper_row(p).strip())
        lines.append("\n</details>\n")

    return "\n".join(lines)



# ─────────────────────────────────────────────
# GitPage Markdown generator (docs/index.md)
# Restored from daily_arxiv.py to_web=True logic
# ─────────────────────────────────────────────

def _paper_row_web(p: Paper) -> str:
    """GitHub Pages row — no <details>, includes back-to-top link"""
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
    today_dot  = datetime.date.today().strftime("%Y.%m.%d")
    today_anchor = f"#updated-on-{today_dot.replace('.', '')}"
    cat_names  = list(config.get("categories", {}).keys())
    cfg        = config.get("settings", {})
    user       = config.get("user_name", "cold-young").replace(" ", "-")   # config compat
    repo       = config.get("repo_name", "robotics-paper-daily")

    lines = []

    # ── Jekyll front matter
    lines.append("---")
    lines.append("layout: default")
    lines.append("---")
    lines.append("")

    # ── Badges (restored from daily_arxiv.py show_badge)
    lines.append(f"[![Contributors][contributors-shield]][contributors-url]")
    lines.append(f"[![Forks][forks-shield]][forks-url]")
    lines.append(f"[![Stargazers][stars-shield]][stars-url]")
    lines.append(f"[![Issues][issues-shield]][issues-url]")
    lines.append("")

    lines.append(f"## Updated on {today_dot}")
    lines.append(f"> Usage instructions: [here](./docs/README.md#usage)")
    lines.append("")

    # ── Per-section body (expanded, no <details>)
    def _section(title: str, papers: list[Paper]) -> list[str]:
        anchor_id = title.lower().replace(" ", "-")
        sec = []
        sec.append(f"## {title}")
        sec.append("")
        sec.append("| Publish Date | Title & Abstract | Authors | Links |")
        sec.append("|:---------|:-----------------------|:---------|:------|")
        for p in papers:
            sec.append(_paper_row_web(p))
        sec.append("")
        sec.append(f"<p align=right>(<a href={today_anchor}>back to top</a>)</p>")
        sec.append("")
        return sec

    # HF Hot Papers section
    hf_papers = sorted(
        [p for p in all_papers.values() if p.hf_rank is not None],
        key=lambda p: p.hf_rank,
    )
    if hf_papers:
        lines += _section("🔥 HuggingFace Hot Papers", hf_papers)

    # Per-category sections
    for cat_name in cat_names:
        cat_papers = sorted(
            [p for p in all_papers.values() if cat_name in p.matched_categories],
            key=lambda p: p.publish_date,
            reverse=True,
        )
        if cat_papers:
            lines += _section(cat_name, cat_papers)

    # ── Badge link definitions (footer)
    lines.append(f"[contributors-shield]: https://img.shields.io/github/contributors/{user}/{repo}.svg?style=for-the-badge")
    lines.append(f"[contributors-url]: https://github.com/{user}/{repo}/graphs/contributors")
    lines.append(f"[forks-shield]: https://img.shields.io/github/forks/{user}/{repo}.svg?style=for-the-badge")
    lines.append(f"[forks-url]: https://github.com/{user}/{repo}/network/members")
    lines.append(f"[stars-shield]: https://img.shields.io/github/stars/{user}/{repo}.svg?style=for-the-badge")
    lines.append(f"[stars-url]: https://github.com/{user}/{repo}/stargazers")
    lines.append(f"[issues-shield]: https://img.shields.io/github/issues/{user}/{repo}.svg?style=for-the-badge")
    lines.append(f"[issues-url]: https://github.com/{user}/{repo}/issues")

    return "\n".join(lines)


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
    print(f"[Step 3] Merging... (max {args.max_per_cat} per category)")
    db = merge_into_db(db, today_papers, max_per_category=args.max_per_cat)
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
        gitpage_md   = generate_gitpage_markdown(display_papers, hf_map, config)
        gitpage_path = Path(args.gitpage)
        gitpage_path.parent.mkdir(parents=True, exist_ok=True)
        gitpage_path.write_text(gitpage_md, encoding="utf-8")
        print(f"✅ GitPage → {gitpage_path}")


if __name__ == "__main__":
    main()
