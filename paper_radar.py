"""
paper_radar.py  ─  Improved robotics paper tracker
────────────────────────────────────────────────────
Improvements over cold-young/robotics_paper_daily:
  1. arXiv ID 기준 cross-category deduplication
  2. HuggingFace Daily Papers 통합
  3. 관련도 점수 기반 정렬
  4. Papers With Code API 통합 (코드 링크 자동 수집)
  5. config.yaml 완전 호환
  6. 📦 누적 DB (docs/papers_db.json)
     → 매일 새 논문이 기존에 쌓이고, 카테고리별 최대 N개 초과 시
       가장 오래된 논문부터 제거 (기본값: 50개/카테고리)

Usage:
    python paper_radar.py                  # 오늘 날짜 기준, DB에 누적
    python paper_radar.py --days 3         # 최근 3일치 수집 후 누적
    python paper_radar.py --reset-db       # DB 초기화 후 새로 수집
    python paper_radar.py --output my.md   # 출력 파일 지정
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
    # Papers With Code 필드 (기존 config의 base_url 복원)
    code_url: str = ""        # GitHub repo URL
    pwc_url: str = ""         # paperswithcode.com 페이지
    framework: str = ""       # "PyTorch" / "TensorFlow" / ""
    # 수집 메타
    matched_keywords: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    hf_rank: Optional[int] = None   # HuggingFace daily rank (1-based)
    score: int = 0                   # 관련도 점수 (정렬용)

    def compute_score(self):
        """키워드 히트 수 + HF 순위 가중치 + 코드 공개 보너스"""
        self.score = len(self.matched_keywords) * 10
        if self.hf_rank is not None:
            self.score += max(0, 30 - self.hf_rank)  # top-1=+29, top-30=+1
        if self.code_url:
            self.score += 5   # 코드 공개 논문 소폭 우선

    def keyword_badges(self) -> str:
        badges = []
        for cat in self.matched_categories:
            badges.append(f"`{cat}`")
        if self.hf_rank is not None:
            badges.append(f"🔥 HF#{self.hf_rank}")
        if self.code_url:
            badges.append("💻 Code")
        return " ".join(badges)

    def author_team(self) -> str:
        """Last Author Team 형식 (기존 포맷 유지)"""
        parts = [a.strip() for a in self.authors.split(",")]
        if len(parts) >= 2:
            return f"{parts[-1]} Team"
        return parts[0]

# ─────────────────────────────────────────────
# Papers With Code fetcher (기존 config base_url 복원)
# ─────────────────────────────────────────────

# config.yaml의 base_url 그대로 사용
PWC_BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"
ARXIV_API    = "https://export.arxiv.org/api/query"  # fallback


def fetch_pwc_by_id(arxiv_id: str) -> dict:
    """
    Papers With Code API로 단일 논문 코드 정보 조회.
    응답 예시:
      {
        "paper": {"id": "2603.09761", "title": "...", ...},
        "repository": {"url": "https://github.com/...", "framework": "PyTorch"},
        "paper_with_code": {"url": "https://paperswithcode.com/paper/..."}
      }
    실패 시 빈 dict 반환.
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
    Paper 객체에 PWC 코드 정보를 인플레이스로 추가.
    score에 코드 존재 보너스도 반영.
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
# 누적 DB (docs/papers_db.json)
# ─────────────────────────────────────────────

DB_DEFAULT_PATH = "docs/papers_db.json"


def _paper_to_dict(p: Paper) -> dict:
    return dataclasses.asdict(p)


def _paper_from_dict(d: dict) -> Paper:
    return Paper(**d)


def load_db(db_path: str = DB_DEFAULT_PATH) -> dict[str, Paper]:
    """
    JSON DB 로드 → {arxiv_id: Paper}
    파일이 없으면 빈 dict 반환.
    """
    path = Path(db_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {aid: _paper_from_dict(d) for aid, d in raw.items()}
    except Exception as e:
        print(f"[WARN] DB 로드 실패 ({db_path}): {e} — 빈 DB로 시작")
        return {}


def save_db(papers: dict[str, Paper], db_path: str = DB_DEFAULT_PATH) -> None:
    """
    {arxiv_id: Paper} → JSON DB 저장
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {aid: _paper_to_dict(p) for aid, p in papers.items()}
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_into_db(
    db: dict[str, Paper],
    new_papers: dict[str, Paper],
    max_per_category: int = 50,
) -> dict[str, Paper]:
    """
    새 논문을 기존 DB에 병합한 뒤, 카테고리별 최대 개수를 초과하면
    가장 오래된 논문(publish_date 기준)을 제거한다.

    규칙:
      - 동일 arxiv_id가 이미 있으면 category/keyword 태그만 업데이트
        (코드 링크, HF rank 등 새 정보도 갱신)
      - 없으면 새로 추가
      - 전체 DB에서 각 카테고리별로 최신 N개만 유지
        (여러 카테고리에 걸친 논문은 각 카테고리 계산에 모두 포함)
    """
    merged = dict(db)  # 기존 DB 복사

    # 1. 새 논문 병합
    for aid, new_p in new_papers.items():
        if aid in merged:
            old_p = merged[aid]
            # 카테고리/키워드 태그 누적
            for cat in new_p.matched_categories:
                if cat not in old_p.matched_categories:
                    old_p.matched_categories.append(cat)
            for kw in new_p.matched_keywords:
                if kw not in old_p.matched_keywords:
                    old_p.matched_keywords.append(kw)
            # 새 정보로 갱신 (코드링크, HF rank 등)
            if new_p.code_url:
                old_p.code_url  = new_p.code_url
                old_p.framework = new_p.framework
            if new_p.pwc_url:
                old_p.pwc_url = new_p.pwc_url
            if new_p.hf_rank is not None:
                old_p.hf_rank = new_p.hf_rank
        else:
            merged[aid] = new_p

    # 2. 카테고리별 최대 개수 적용 (오래된 것 제거)
    #    카테고리 목록 수집 (HF-Hot 제외: 매일 바뀌므로 별도 관리)
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
        # 날짜 오름차순 정렬 → 초과분(오래된 것)을 제거 후보로
        cat_papers_sorted = sorted(cat_papers, key=lambda p: p.publish_date)
        overflow = len(cat_papers) - max_per_category
        for old_p in cat_papers_sorted[:overflow]:
            # 다른 카테고리에도 속한 논문은 해당 카테고리 태그만 제거
            old_p.matched_categories = [
                c for c in old_p.matched_categories if c != cat
            ]
            # 아무 카테고리도 없어진 논문은 DB에서 완전 삭제 대상
            if not old_p.matched_categories:
                papers_to_remove.add(old_p.arxiv_id)

    for aid in papers_to_remove:
        del merged[aid]

    return merged


def get_display_papers(
    db: dict[str, Paper],
    hf_map: dict[str, int],
) -> dict[str, Paper]:
    """
    DB에서 표시용 논문 dict를 반환.
    HF rank는 오늘 기준으로 갱신 (누적 HF rank는 의미없음).
    """
    display = {}
    for aid, p in db.items():
        # 새 Paper 복사 (원본 DB 불변 보장)
        dp = Paper(**dataclasses.asdict(p))
        # HF rank는 오늘 것으로 덮어쓰기 (또는 초기화)
        dp.hf_rank = hf_map.get(aid, None)
        dp.compute_score()
        display[aid] = dp
    return display

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

    # 3. HF rank 부여
    for aid, paper in all_papers.items():
        if aid in hf_map:
            paper.hf_rank = hf_map[aid]

    # 4. HF 핫 논문 중 아직 없는 것도 추가 (arXiv에서 개별 fetch)
    for hf_id, rank in hf_map.items():
        if hf_id not in all_papers:
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

    # 5. Papers With Code 코드 링크 보강 (config base_url 복원)
    #    rate limit 방지: 0.5초 간격
    print(f"\n[PWC] Enriching {len(all_papers)} papers with code links...")
    pwc_count = 0
    for i, (aid, paper) in enumerate(all_papers.items()):
        time.sleep(0.5)
        enrich_with_pwc(paper)
        if paper.code_url:
            pwc_count += 1
        if (i + 1) % 20 == 0:
            print(f"  → {i+1}/{len(all_papers)} processed, {pwc_count} with code")
    print(f"[PWC] Done. {pwc_count}/{len(all_papers)} papers have code links.")

    # 6. 최종 score 계산 (코드 보너스 포함)
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
    """기존 README 테이블 포맷 + 개선 (PWC 코드 링크 포함)"""
    links = f"[ArXiv]({p.arxiv_url})"
    if p.code_url:
        links += f" / [Code]({p.code_url})"      # GitHub 링크 우선
    elif p.project_url:
        links += f" / [Web]({p.project_url})"    # 없으면 project page
    if p.pwc_url:
        links += f" / [PWC]({p.pwc_url})"        # Papers With Code 페이지

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
            if p.code_url:
                links += f" / [Code]({p.code_url})"
            elif p.project_url:
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
# GitPage Markdown generator (docs/index.md)
# 기존 daily_arxiv.py의 to_web=True 로직 복원
# ─────────────────────────────────────────────

def _paper_row_web(p: Paper) -> str:
    """GitHub Pages용 행 — <details> 미사용, back-to-top 링크 포함"""
    links = f"[ArXiv]({p.arxiv_url})"
    if p.code_url:
        links += f" / [Code]({p.code_url})"
    elif p.project_url:
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
    GitHub Pages 호환 마크다운 생성 (docs/index.md).
    기존 daily_arxiv.py의 to_web=True 포맷 복원:
      - Jekyll front matter (layout: default)
      - <details> 미사용 → 모든 내용 펼쳐진 상태
      - 각 섹션 하단에 back-to-top 링크
      - 배지 (contributors / forks / stars / issues)
    """
    today_dot  = datetime.date.today().strftime("%Y.%m.%d")
    today_anchor = f"#updated-on-{today_dot.replace('.', '')}"
    cat_names  = list(config.get("categories", {}).keys())
    cfg        = config.get("settings", {})
    user       = config.get("user_name", "cold-young").replace(" ", "-")   # config 호환
    repo       = config.get("repo_name", "robotics-paper-daily")

    lines = []

    # ── Jekyll front matter
    lines.append("---")
    lines.append("layout: default")
    lines.append("---")
    lines.append("")

    # ── 배지 (기존 daily_arxiv.py show_badge 복원)
    lines.append(f"[![Contributors][contributors-shield]][contributors-url]")
    lines.append(f"[![Forks][forks-shield]][forks-url]")
    lines.append(f"[![Stargazers][stars-shield]][stars-url]")
    lines.append(f"[![Issues][issues-shield]][issues-url]")
    lines.append("")

    lines.append(f"## Updated on {today_dot}")
    lines.append(f"> Usage instructions: [here](./docs/README.md#usage)")
    lines.append("")

    # ── 섹션별 본문 (펼쳐진 형태)
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

    # HF Hot Papers 섹션
    hf_papers = sorted(
        [p for p in all_papers.values() if p.hf_rank is not None],
        key=lambda p: p.hf_rank,
    )
    if hf_papers:
        lines += _section("🔥 HuggingFace Hot Papers", hf_papers)

    # All Papers 섹션
    all_sorted = sorted(
        all_papers.values(),
        key=lambda p: (p.publish_date, -p.score),
        reverse=True,
    )
    lines += _section("📊 All Papers (Deduplicated)", all_sorted)

    # 카테고리별 섹션
    for cat_name in cat_names:
        cat_papers = sorted(
            [p for p in all_papers.values() if cat_name in p.matched_categories],
            key=lambda p: p.publish_date,
            reverse=True,
        )
        if cat_papers:
            lines += _section(cat_name, cat_papers)

    # ── 배지 링크 정의 (하단)
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
    parser.add_argument("--db",         default="docs/papers_db.json", help="누적 DB JSON 경로")
    parser.add_argument("--days",       type=int, default=1,         help="오늘부터 며칠 치 수집")
    parser.add_argument("--max-per-cat",type=int, default=50,        help="카테고리당 최대 논문 수")
    parser.add_argument("--no-gitpage", action="store_true",         help="GitPage 생성 건너뜀")
    parser.add_argument("--reset-db",   action="store_true",         help="DB 초기화 후 새로 수집")
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Step 1: 오늘 새 논문 수집
    print("\n[Step 1] 새 논문 수집 중...")
    today_papers, hf_map = collect_papers(config, days_back=args.days)

    # ── Step 2: DB 로드 (또는 초기화)
    if args.reset_db:
        print("[Step 2] DB 초기화 (--reset-db 플래그)")
        db = {}
    else:
        print(f"[Step 2] DB 로드: {args.db}")
        db = load_db(args.db)
        print(f"         기존 DB: {len(db)}개 논문")

    # ── Step 3: 새 논문을 DB에 병합 + 초과분 정리
    print(f"[Step 3] 병합 중... (카테고리당 최대 {args.max_per_cat}개)")
    db = merge_into_db(db, today_papers, max_per_category=args.max_per_cat)
    print(f"         병합 후 DB: {len(db)}개 논문")

    # ── Step 4: DB 저장
    save_db(db, args.db)
    print(f"[Step 4] DB 저장 완료: {args.db}")

    # ── Step 5: 표시용 논문 준비 (HF rank 오늘 기준으로 갱신)
    display_papers = get_display_papers(db, hf_map)

    # ── 통계 출력
    total      = len(display_papers)
    hf_count   = sum(1 for p in display_papers.values() if p.hf_rank is not None)
    multi_cat  = sum(1 for p in display_papers.values() if len(p.matched_categories) > 1)
    code_count = sum(1 for p in display_papers.values() if p.code_url)
    cat_stats  = {}
    for p in display_papers.values():
        for cat in p.matched_categories:
            if cat != "HF-Hot":
                cat_stats[cat] = cat_stats.get(cat, 0) + 1

    print(f"\n[Stats] 누적 논문 총 {total}개")
    print(f"[Stats] HF hot papers     : {hf_count}")
    print(f"[Stats] 멀티카테고리 논문 : {multi_cat}")
    print(f"[Stats] 코드 링크 있음    : {code_count}")
    for cat, cnt in sorted(cat_stats.items()):
        bar = "█" * int(cnt / args.max_per_cat * 20)
        print(f"[Stats]   {cat:<18} {cnt:>3}/{args.max_per_cat}  {bar}")

    # ── README.md 생성
    readme_md = generate_markdown(display_papers, hf_map, config)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(readme_md, encoding="utf-8")
    print(f"\n✅ README  → {out_path}")

    # ── docs/index.md 생성 (GitPage)
    if not args.no_gitpage:
        gitpage_md   = generate_gitpage_markdown(display_papers, hf_map, config)
        gitpage_path = Path(args.gitpage)
        gitpage_path.parent.mkdir(parents=True, exist_ok=True)
        gitpage_path.write_text(gitpage_md, encoding="utf-8")
        print(f"✅ GitPage → {gitpage_path}")


if __name__ == "__main__":
    main()
