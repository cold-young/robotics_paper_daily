"""
paper_radar.py  ─  Improved robotics paper tracker
────────────────────────────────────────────────────
Improvements over cold-young/robotics_paper_daily:
  1. arXiv ID 기준 cross-category deduplication
     → 같은 논문이 여러 키워드에 걸릴 때 한 번만 표시,
       매칭된 키워드 태그를 모두 병기
  2. HuggingFace Daily Papers 통합
     → 로봇/ML 핫 논문을 별도 섹션으로 상단에 표시
  3. 관련도 점수 (키워드 히트 수 + HF 순위) 기반 정렬
  4. config.yaml 완전 호환

Usage:
    python paper_radar.py                  # 오늘 날짜 기준
    python paper_radar.py --days 3         # 최근 3일치
    python paper_radar.py --output my.md   # 출력 파일 지정
"""

import re
import time
import argparse
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml  # pip install pyyaml

# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: str          # "First Author et al."
    publish_date: str     # "YYYY-MM-DD"
    arxiv_url: str
    project_url: str = ""
    # 수집 메타
    matched_keywords: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    hf_rank: Optional[int] = None   # HuggingFace daily rank (1-based)
    score: int = 0                   # 관련도 점수 (정렬용)

    def compute_score(self):
        """키워드 히트 수 + HF 순위 가중치"""
        self.score = len(self.matched_keywords) * 10
        if self.hf_rank is not None:
            self.score += max(0, 30 - self.hf_rank)  # top-1=+29, top-30=+1

    def keyword_badges(self) -> str:
        badges = []
        for cat in self.matched_categories:
            badges.append(f"`{cat}`")
        if self.hf_rank is not None:
            badges.append(f"🔥 HF#{self.hf_rank}")
        return " ".join(badges)

    def author_team(self) -> str:
        """Last Author Team 형식 (기존 포맷 유지)"""
        parts = [a.strip() for a in self.authors.split(",")]
        if len(parts) >= 2:
            return f"{parts[-1]} Team"
        return parts[0]

# ─────────────────────────────────────────────
# arXiv fetcher
# ─────────────────────────────────────────────

ARXIV_API = "https://export.arxiv.org/api/query"

def fetch_arxiv(keywords: list[str], max_results: int = 50,
                days_back: int = 1) -> list[dict]:
    """
    키워드 리스트로 arXiv 검색 → raw result list 반환
    """
    query_parts = [f'ti:"{kw}" OR abs:"{kw}"' for kw in keywords]
    query = " OR ".join(query_parts)

    params = urllib.parse.urlencode({
        "search_query": query,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "PaperRadar/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)

    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)
    results = []
    for entry in root.findall("atom:entry", ns):
        published = entry.find("atom:published", ns).text[:10]
        if datetime.date.fromisoformat(published) < cutoff:
            continue

        raw_id = entry.find("atom:id", ns).text
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]

        title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
        abstract = entry.find("atom:summary", ns).text.replace("\n", " ").strip()

        authors_raw = entry.findall("atom:author", ns)
        author_names = [a.find("atom:name", ns).text for a in authors_raw]
        authors_str = ", ".join(author_names)

        # project page 링크 (comment 필드)
        project_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                continue
            href = link.get("href", "")
            if href.startswith("http") and "arxiv" not in href:
                project_url = href
                break

        results.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors_str,
            "publish_date": published,
            "arxiv_url": f"http://arxiv.org/abs/{arxiv_id}",
            "project_url": project_url,
        })
    return results


# ─────────────────────────────────────────────
# HuggingFace Daily Papers fetcher
# ─────────────────────────────────────────────

HF_DAILY_API = "https://huggingface.co/api/daily_papers"

def fetch_hf_daily(date_str: Optional[str] = None,
                   limit: int = 30) -> dict[str, int]:
    """
    HuggingFace Daily Papers API → {arxiv_id: rank} dict 반환
    date_str 예: "2024-03-12"  (None이면 오늘)
    """
    params = {"limit": limit}
    if date_str:
        params["date"] = date_str

    url = HF_DAILY_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "PaperRadar/1.0",
        "Accept": "application/json",
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[WARN] HuggingFace API 접근 실패: {e}")
        return {}

    hf_map = {}
    for rank, item in enumerate(data, start=1):
        paper = item.get("paper", {})
        pid = paper.get("id", "")          # e.g. "2403.12345"
        if pid:
            hf_map[pid] = rank
    return hf_map


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    기존 config.yaml 포맷 호환:
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

def collect_papers(config: dict, days_back: int = 1) -> tuple[
    dict[str, Paper],   # all papers (deduped), keyed by arxiv_id
    dict[str, int],     # hf_map
]:
    categories: dict[str, list[str]] = {}
    for cat_name, cat_cfg in config.get("categories", {}).items():
        categories[cat_name] = cat_cfg.get("keywords", [])

    # 1. HuggingFace hot papers
    today = datetime.date.today().isoformat()
    hf_map = fetch_hf_daily(date_str=today)
    print(f"[HF] {len(hf_map)} daily papers fetched")

    # 2. arXiv per category
    all_papers: dict[str, Paper] = {}

    for cat_name, keywords in categories.items():
        print(f"[arXiv] Fetching category '{cat_name}' with {len(keywords)} keywords...")
        time.sleep(3)  # arXiv API rate limit 준수

        raw = fetch_arxiv(keywords, max_results=100, days_back=days_back)
        print(f"  → {len(raw)} papers before dedup")

        for r in raw:
            aid = r["arxiv_id"]

            # 키워드 매칭 확인 (어떤 키워드에 걸렸는지)
            text = (r["title"] + " " + r["abstract"]).lower()
            matched = [kw for kw in keywords if kw.lower() in text]

            if aid not in all_papers:
                all_papers[aid] = Paper(
                    arxiv_id=aid,
                    title=r["title"],
                    abstract=r["abstract"],
                    authors=r["authors"],
                    publish_date=r["publish_date"],
                    arxiv_url=r["arxiv_url"],
                    project_url=r["project_url"],
                )

            p = all_papers[aid]
            # 새 카테고리/키워드 추가 (중복 없이)
            if cat_name not in p.matched_categories:
                p.matched_categories.append(cat_name)
            for kw in matched:
                if kw not in p.matched_keywords:
                    p.matched_keywords.append(kw)

    # 3. HF rank 부여 + score 계산
    for aid, paper in all_papers.items():
        if aid in hf_map:
            paper.hf_rank = hf_map[aid]

    # 4. HF 핫 논문 중 아직 없는 것도 추가 (arXiv에서 개별 fetch)
    for hf_id, rank in hf_map.items():
        if hf_id not in all_papers:
            # HF 핫 논문은 arXiv에서 직접 가져옴
            try:
                time.sleep(1)
                raw = fetch_arxiv_by_id(hf_id)
                if raw:
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
                    )
                    all_papers[hf_id] = p
            except Exception as e:
                print(f"[WARN] HF paper {hf_id} fetch failed: {e}")

    for p in all_papers.values():
        p.compute_score()

    return all_papers, hf_map


def fetch_arxiv_by_id(arxiv_id: str) -> Optional[dict]:
    """특정 arXiv ID 논문 직접 조회"""
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
    """기존 README 테이블 포맷 + 개선"""
    links = f"[ArXiv]({p.arxiv_url})"
    if p.project_url:
        links += f" / [Web]({p.project_url})"

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

    # ── 목차
    lines.append("## Table of Contents\n")
    lines.append("1. [🔥 HuggingFace Hot Papers](#-huggingface-hot-papers)")
    lines.append("2. [📊 All Papers (Deduplicated)](#-all-papers-deduplicated)")
    for i, cat in enumerate(cat_names, start=3):
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
            links = f"[ArXiv]({p.arxiv_url})"
            if p.project_url:
                links += f" / [Web]({p.project_url})"
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
        lines.append("*오늘의 HuggingFace 핫 논문이 없거나 로봇 관련 항목이 없습니다.*\n")

    # ── Section 2: All (deduplicated, score-sorted)
    lines.append("## 📊 All Papers (Deduplicated)\n")
    lines.append(
        f"> 총 **{len(all_papers)}개** 논문 (키워드 중복 제거됨). "
        "동일 논문이 여러 카테고리에 해당하는 경우 배지로 표시됩니다.\n"
    )

    all_sorted = sorted(all_papers.values(),
                        key=lambda p: (-p.score, p.publish_date),
                        reverse=False)
    all_sorted = sorted(all_papers.values(),
                        key=lambda p: (p.publish_date, -p.score),
                        reverse=True)

    lines.append("<details><summary><b>All Papers (Click to expand)</b></summary>\n")
    lines.append("| Publish Date | Title & Abstract | Authors | Links |")
    lines.append("| --- | --- | --- | --- |")
    for p in all_sorted:
        lines.append(_paper_row(p).strip())
    lines.append("\n</details>\n")

    # ── Section 3+: Per-category (arXiv-only, for reference)
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
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaperRadar: robotics paper tracker")
    parser.add_argument("--config", default="config.yaml", help="config YAML path")
    parser.add_argument("--output", default="README.md", help="output markdown path")
    parser.add_argument("--days", type=int, default=1,
                        help="look back N days (default: 1)")
    args = parser.parse_args()

    config = load_config(args.config)
    all_papers, hf_map = collect_papers(config, days_back=args.days)

    print(f"\n[Result] Total unique papers: {len(all_papers)}")
    print(f"[Result] HF hot papers found: {len(hf_map)}")
    hf_in_result = sum(1 for p in all_papers.values() if p.hf_rank is not None)
    multi_cat = sum(1 for p in all_papers.values() if len(p.matched_categories) > 1)
    print(f"[Result] Papers with HF rank: {hf_in_result}")
    print(f"[Result] Papers in multiple categories (would have been duplicated): {multi_cat}")

    md = generate_markdown(all_papers, hf_map, config)

    out_path = Path(args.output)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n✅ Written to {out_path}")


if __name__ == "__main__":
    main()
