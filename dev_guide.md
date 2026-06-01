# 개발 계획서: 학회 논문 스크랩 기능 추가

> **대상 레포:** `cold-young/robotics_paper_daily`
> **목표:** 기존 arXiv/HuggingFace 수집에 더해 CoRL · RSS · NeurIPS accepted papers를 수집한다.
> **작성 기준:** 사용자 확정 사항 — (1) CoRL/RSS/NeurIPS 우선, (2) 기존 로보틱스 키워드로 필터링, (3) 학회는 월 1회 갱신, arXiv 레포와 통합(별도 레포 X), 워크플로우만 분리.

---

## 0. 핵심 설계 결정 (Agent는 이 전제를 지킬 것)

1. **통합 유지, 워크플로우 분리.** 학회 fetcher는 `paper_radar.py`의 기존 소스 추상화(arXiv/HF → 공통 `Paper` → DB 병합 → README 생성)에 새 소스로 합류한다. 별도 레포를 만들지 않는다. 단, 실행 스케줄은 분리한다: arXiv는 기존 cron(12h), 학회는 신규 cron(월 1회).

2. **공통 데이터 모델 재사용.** 학회 논문도 기존 `Paper` dataclass로 표현한다. 새 필드만 최소 추가한다(아래 1.2 참조).

3. **ID 정규화가 최우선 난제.** 현재 DB는 `arxiv_id`를 primary key로 사용한다. 학회 논문은 arXiv ID가 없을 수 있으므로(OpenReview ID만 존재), 범용 `paper_id` 키 체계로 정규화해야 한다. 이 작업을 하위 호환되게 처리하는 것이 1순위.

4. **기존 동작 무손상.** arXiv/HF 파이프라인의 현재 출력(README 포맷, papers_db.json 스키마)은 깨지지 않아야 한다. 학회 기능은 가산적(additive)이어야 한다.

---

## 1. 데이터 소스 현황 (조사 완료)

### 1.1 학회별 접근 방식

| 학회 | 호스팅 | 접근 방법 | 난이도 |
|---|---|---|---|
| **NeurIPS** | OpenReview | `openreview-py` 클라이언트, venue=`NeurIPS.cc/{year}/Conference` | 낮음 |
| **CoRL** | OpenReview | `openreview-py`, venue=`robot-learning.org/CoRL/{year}/Conference` | 낮음 |
| **RSS** | OpenReview (최근 연도) | `openreview-py`, venue=`roboticsfoundation.org/RSS/{year}/Conference` | 낮음~중간 |
| (보류) ICLR/ICML | OpenReview | 동일 패턴 | 낮음 |
| (보류) CVPR | CVF Open Access | HTML 스크래핑 | 높음 |

> **주의:** OpenReview venue ID 문자열은 연도/학회마다 다를 수 있다. Agent는 구현 중 실제 venue ID를 `openreview-py`의 `client.get_group` 또는 venue 목록 조회로 **검증**해야 한다. 위 표의 ID는 출발점이며, 연도별로 `/Conference` 접미사나 그룹 경로가 달라질 수 있다.

### 1.2 OpenReview API v2 사용 패턴

```python
import openreview

# API v2 client (인증 불필요 — accepted papers는 public)
client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

# accepted papers 조회 — venueid 기반 (get_notes 사용, get_all_notes는 limit 미지원)
notes = client.get_notes(content={"venueid": "robot-learning.org/CoRL/2024/Conference"}, limit=1000)
# 전체를 가져오려면 openreview.tools.iterget_notes 또는 offset 페이지네이션 사용

for note in notes:
    title = note.content["title"]["value"]      # v2는 {"value": ...} 래핑
    abstract = note.content["abstract"]["value"]
    authors = note.content["authors"]["value"]   # list[str]
    pdf_url = note.content.get("pdf", {}).get("value", "")
    # arXiv ID는 없을 수 있음 → OpenReview note.id 사용
```

> **검증된 API 사용법 (실측):**
> - `client.get_notes(content={"venueid": ...}, limit=N)` 사용. `get_all_notes`는 `limit` 인자를 받지 않음.
> - 전체 수집은 `openreview.tools.iterget_notes(client, content={"venueid": ...})`로 페이지네이션.
>
> **⚠️ venue ID 검증은 Agent가 GitHub Actions에서 수행해야 함.** 개발 컨테이너/일부 클라우드 IP는 OpenReview가 "Host not in allowlist"로 차단한다(실측 확인). 따라서 위 venue_id 문자열의 정확성은 **Actions runner 환경에서** 실제 호출로 검증할 것. 로컬에서 차단되면 venue ID가 틀린 게 아니라 IP 문제일 수 있으니 혼동하지 말 것.

---

## 2. 구현 작업 분해 (순서대로)

### Task 1 — ID 정규화 (선행 필수, 하위 호환)

**문제:** DB·dedup·README가 전부 `arxiv_id`에 묶여 있다.

**해결:**
- `Paper` dataclass에 필드 추가:
  - `paper_id: str` — 범용 primary key. arXiv 논문이면 `"arxiv:2605.12345"`, 학회 논문이면 `"openreview:AbC123"`.
  - `source: str` — `"arxiv"` | `"hf"` | `"corl"` | `"rss"` | `"neurips"`.
  - `venue: str = ""` — 표시용 (예: `"CoRL 2024"`).
- `paper_id` 생성 헬퍼 `make_paper_id(source, raw_id) -> str` 작성.
- **하위 호환:** 기존 `papers_db.json`은 `arxiv_id`만 갖고 있다. 로드 시 `paper_id`가 없으면 `f"arxiv:{arxiv_id}"`로 채우는 마이그레이션 로직을 `_paper_from_dict`에 추가.
- DB dict 키를 `arxiv_id` → `paper_id`로 전환. `load_db`/`save_db`/`merge_into_db`의 키 사용처를 모두 수정.

**검증:** 기존 `papers_db.json`을 로드 → 저장 → 다시 로드해서 논문 수·내용이 동일한지 확인 (마이그레이션 무손실).

---

### Task 2 — 학회 fetcher 모듈 작성

**신규 파일:** `conference_fetch.py` (paper_radar.py를 비대하게 만들지 않기 위해 분리)

**함수 시그니처:**
```python
def fetch_openreview_venue(
    venue_id: str,          # "robot-learning.org/CoRL/2024/Conference"
    venue_label: str,       # "CoRL 2024" (표시용)
    source: str,            # "corl" | "rss" | "neurips"
    keywords: list[str],    # 로보틱스 키워드 (필터용)
    max_results: int = 0,   # 0이면 무제한
) -> list[dict]:
    """OpenReview에서 accepted papers를 가져와 keyword 필터링 후
    paper_radar의 raw dict 포맷(title/abstract/authors/...)으로 반환."""
```

**구현 요점:**
- `openreview-py`의 `get_all_notes`로 accepted papers 조회.
- 키워드 필터: `(title + abstract).lower()`에 키워드가 하나라도 포함되면 채택 (기존 arXiv 로직과 동일).
- arXiv ID 매칭 시도: OpenReview note에 arXiv 링크가 있으면 추출해 dedup에 활용(선택). 없으면 `openreview:{note.id}`.
- `publish_date`: accepted 논문은 정확한 날짜가 모호하므로 **학회 연도 기준** 고정값 사용 (예: `"{year}-01-01"`) 또는 note의 `cdate`(생성일). README 정렬을 위해 일관성만 있으면 됨.
- v1/v2 content 래핑 차이 방어 처리.
- 네트워크 실패 시 빈 리스트 반환 + `[ERROR]` 로그 (arXiv fetcher와 동일한 가시적 에러 정책).

**검증:** CoRL 2024 단일 venue로 호출 → 논문 N편 반환되고 각 dict에 title/abstract/authors가 채워지는지 확인. 키워드 필터 전후 개수 로그.

---

### Task 3 — config.yaml 확장

**추가 섹션:**
```yaml
conferences:
  enabled: true
  # 로보틱스 키워드는 기존 categories의 keywords를 재사용한다.
  # (별도 명시 안 하면 모든 categories 키워드를 합쳐서 필터)
  venues:
    - source: corl
      label: "CoRL 2024"
      venue_id: "robot-learning.org/CoRL/2024/Conference"
    - source: rss
      label: "RSS 2024"
      venue_id: "roboticsfoundation.org/RSS/2024/Conference"
    - source: neurips
      label: "NeurIPS 2024"
      venue_id: "NeurIPS.cc/2024/Conference"
  max_results_per_venue: 0   # 0 = 무제한 (필터로 충분히 줄어듦)
```

> Agent는 위 `venue_id`들을 **실제로 호출해 검증**하고, 틀리면 OpenReview venue 목록에서 올바른 ID를 찾아 교정한다.

**검증:** config 파싱 후 venue 리스트가 올바르게 읽히는지 단위 확인.

---

### Task 4 — collect_papers에 학회 소스 통합

**수정 파일:** `paper_radar.py`의 `collect_papers()`

- 함수 시그니처에 `include_conferences: bool = False` 추가.
- arXiv/HF 수집 이후, `include_conferences`가 True면 `conference_fetch.fetch_openreview_venue`를 venue별로 호출.
- 반환된 논문을 기존 `all_papers` dict에 병합 (paper_id 키 기준 dedup).
  - 키워드 매칭/카테고리 부여 로직은 arXiv와 공유 (헬퍼로 추출해 재사용).
  - 학회 논문에는 `matched_categories`에 매칭된 카테고리 + `source` 표시.
- **PWC enrichment는 학회 논문에 스킵** (arXiv ID 없으면 PWC 조회 불가/무의미). arXiv ID가 매칭된 경우만 선택적으로.

**검증:** `include_conferences=True`로 전체 collect 실행 → arXiv 논문 수는 기존과 동일, 학회 논문이 추가로 들어오는지 확인. DB 저장/로드 무손상 재확인.

---

### Task 5 — README/GitPage 출력에 학회 표시

**수정 파일:** `paper_radar.py`의 `generate_markdown`, `_paper_row`, `keyword_badges`

- `Paper.keyword_badges()`에 venue 배지 추가: source가 학회면 `` `📚 CoRL 2024` `` 형태.
- 링크 처리: 학회 논문은 arXiv URL 대신 OpenReview/PDF URL을 링크. `_paper_row`에서 `arxiv_url`이 없으면 `project_url`/OpenReview URL fallback.
- (선택) README에 학회별 섹션을 추가하거나, 기존 카테고리 섹션 안에 배지로만 구분. **권장:** 기존 카테고리 섹션 유지 + venue 배지로 구분 (구조 변경 최소화).

**검증:** 학회 논문이 포함된 README를 생성 → 배지·링크가 깨지지 않고, 기존 arXiv row 포맷도 그대로인지 육안 확인.

---

### Task 6 — main()에 모드 분기 + 신규 워크플로우

**수정 파일:** `paper_radar.py`의 `main()`
- CLI 인자 `--conferences` 추가. 있으면 `collect_papers(include_conferences=True)` 호출, 없으면 기존 arXiv-only 동작.
- 두 모드 모두 같은 DB에 병합 (월 1회 학회 실행이 DB에 학회 논문을 누적, 매일 arXiv 실행이 arXiv 논문을 누적).

**신규 파일:** `.github/workflows/conference_radar.yml`
```yaml
name: Conference Radar (Monthly)
on:
  schedule:
    - cron: "0 0 1 * *"      # 매월 1일 UTC 00:00
  workflow_dispatch:
    inputs:
      year:
        description: "수집 연도 (비우면 config 기본값)"
        required: false
permissions:
  contents: write
jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt openreview-py
      - run: python paper_radar.py --conferences
      - name: Commit DB + README
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m "📚 Conference update $(date +%Y.%m.%d)" || echo "No changes"
          git push || echo "Nothing to push"
```

**검증:** workflow_dispatch로 수동 실행 → 학회 논문이 DB/README에 반영되고 커밋되는지 확인.

---

### Task 7 — requirements.txt 갱신

```
requests
arxiv>=2.1.0
pyyaml
openreview-py>=1.40.0
```

---

## 3. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| OpenReview venue ID가 표의 값과 다름 | fetch 0건 | Task 2에서 실제 호출로 venue ID 검증, 틀리면 그룹 조회로 교정 |
| API v1/v2 content 래핑 차이 | KeyError | 두 포맷 모두 처리하는 접근자 헬퍼 작성 |
| 학회 논문 arXiv ID 부재 | dedup/PWC 깨짐 | Task 1의 paper_id 정규화로 해결, PWC는 스킵 |
| 기존 papers_db.json 스키마 변경 | 다운스트림(podcast/telegram) 파싱 깨짐 | README **출력 포맷은 유지**, 새 필드는 추가만. podcast 파서가 읽는 테이블 구조 불변 보장 |
| 학회 논문 대량 유입으로 README 비대 | 가독성 저하 | 키워드 필터 + venue별 상한(필요시) |

> **다운스트림 영향 체크 (중요):** `robotics_paper_podcast`가 이 README의 테이블을 정규식으로 파싱한다. Agent는 Task 5에서 **기존 테이블 row 포맷(`| 날짜 | **제목** <details>...| 저자 | [ArXiv](url) |`)을 깨지 않아야** 한다. 학회 논문도 같은 row 구조를 따르되 링크만 OpenReview로, 배지만 추가하는 방식으로 구현.

---

## 4. 완료 기준 (Definition of Done)

1. `python paper_radar.py` (기존) → arXiv/HF만 수집, 출력·DB 스키마 기존과 동일.
2. `python paper_radar.py --conferences` → CoRL/RSS/NeurIPS accepted papers가 로보틱스 키워드 필터를 거쳐 DB에 추가됨.
3. 기존 `papers_db.json` 로드 → 마이그레이션 → 저장 시 무손실.
4. README에 학회 논문이 venue 배지와 함께 표시되고, 기존 arXiv row 포맷 불변.
5. `conference_radar.yml`이 월 1회 + 수동 실행으로 동작.
6. `robotics_paper_podcast`의 README 파서가 변경된 README를 정상 파싱 (회귀 테스트).

---

## 5. 권장 진행 순서

```
Task 1 (ID 정규화) → 검증
   ↓
Task 2 (학회 fetcher 단독) → CoRL 2024로 검증
   ↓
Task 3 (config) → Task 4 (collect 통합) → 검증
   ↓
Task 5 (README 출력) → podcast 파서 회귀 테스트
   ↓
Task 6 (워크플로우) → 수동 실행 검증
   ↓
Task 7 (requirements)
```

각 Task는 독립 커밋. Task 1·4 이후에는 반드시 **기존 arXiv 동작 회귀 테스트**를 먼저 통과시킬 것.