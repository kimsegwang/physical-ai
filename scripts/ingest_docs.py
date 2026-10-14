"""프로젝트 문서를 청킹·임베딩해서 Supabase(pgvector)에 적재한다.

예) python scripts/ingest_docs.py            (바뀐 문서만 다시 적재)
    python scripts/ingest_docs.py --dry-run  (API 호출 없이 청크 개수·길이만 확인)
    python scripts/ingest_docs.py --force    (전부 다시 적재)

필요한 환경 변수 (.env 또는 셸에서 설정, 값은 코드에 적지 않는다):
    GEMINI_API_KEY        Gemini 임베딩 API 키
    SUPABASE_URL          https://<project_ref>.supabase.co
    SUPABASE_SECRET_KEY   Supabase secret 키 (sb_secret_...). RLS를 우회하므로 서버·로컬에서만 사용
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subway_rag.docs_chunker import (  # noqa: E402
    Chunk,
    chunk_markdown,
    document_title,
    html_to_markdown,
)

# 검색 대상 문서 (저장소 기준 경로). design/ 시안, CLAUDE.md(개발 규칙),
# report/reference/(편의점 무인계산대 과제 등 이 프로젝트와 무관한 참고 자료)는 제외한다.
SOURCES = [
    "README.md",
    "PRD/PRD.md",
    "report/preprocess.md",
    "report/model_plan.md",
    "report/vision_models.md",
    "report/checklist.md",
    "report/hardware.md",
    "index.html",
    "docs.html",
]

# 첫 '# ' 제목이 문서 제목으로 적절하지 않은 경우 (템플릿 문구, 히어로 문구 등)
TITLE_OVERRIDES = {
    "index.html": "FlowSight 홈페이지",
    "docs.html": "FlowSight 추가 문서: 비전 모델 비교",
}

EMBED_MODEL ="gemini-embedding-2"
EMBED_DIM = 768  # DB의 vector(768)과 맞춘다
EMBED_BATCH = 50
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_dotenv(path: Path) -> None:
    """.env가 있으면 아직 설정되지 않은 환경 변수만 채운다 (값을 출력하지 않는다)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_source(rel: str) -> tuple[str, str]:
    """(제목, 마크다운 본문)을 돌려준다. HTML은 본문 텍스트만 뽑는다."""
    raw = (ROOT / rel).read_text(encoding="utf-8")
    text = html_to_markdown(raw) if rel.endswith(".html") else raw
    return TITLE_OVERRIDES.get(rel) or document_title(text, fallback=rel), text


def embed_text(title: str, chunk: Chunk) -> str:
    """gemini-embedding-2 권장 문서 형식: 'title: ... | text: ...'."""
    where = f"{title} > {chunk.heading}" if chunk.heading else title
    return f"title: {where} | text: {chunk.content}"


def http_json(method: str, url: str, headers: dict[str, str], body=None):  # noqa: ANN001
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read()
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503) and attempt < 5:
                # Gemini 무료 등급은 분당 요청 한도가 있다. 응답에 적힌 대기 시간만큼 기다린 뒤 다시 시도
                m = re.search(r"retry in ([\d.]+)s", detail)
                wait = float(m.group(1)) + 2 if m else 2 ** attempt * 2
                print(f"  요청 한도 또는 일시 오류 ({e.code}), {wait:.0f}초 기다린 뒤 다시 시도합니다")
                time.sleep(wait)
                continue
            raise RuntimeError(f"{method} {url.split('?')[0]} 실패 ({e.code}): {detail[:500]}") from None


class Gemini:
    def __init__(self, api_key: str) -> None:
        self.headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            body = {
                "requests": [
                    {
                        "model": f"models/{EMBED_MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "outputDimensionality": EMBED_DIM,
                    }
                    for t in batch
                ]
            }
            res = http_json(
                "POST", f"{GEMINI_BASE}/models/{EMBED_MODEL}:batchEmbedContents", self.headers, body
            )
            vectors = [e["values"] for e in res["embeddings"]]
            if any(len(v) != EMBED_DIM for v in vectors):
                raise RuntimeError(f"임베딩 차원이 {EMBED_DIM}이 아닙니다")
            out.extend(vectors)
        return out


class Supabase:
    """PostgREST를 secret 키로 호출한다. 새 secret 키는 apikey 헤더로만 보낸다."""

    def __init__(self, url: str, secret_key: str) -> None:
        self.rest = url.rstrip("/") + "/rest/v1"
        self.headers = {"Content-Type": "application/json", "apikey": secret_key}

    def _h(self, prefer: str | None = None) -> dict[str, str]:
        return {**self.headers, "Prefer": prefer} if prefer else self.headers

    def list_documents(self) -> dict[str, dict]:
        rows = http_json("GET", f"{self.rest}/documents?select=id,source,content_hash", self._h())
        return {r["source"]: r for r in rows}

    def upsert_document(self, source: str, title: str) -> int:
        rows = http_json(
            "POST",
            f"{self.rest}/documents?on_conflict=source",
            self._h("resolution=merge-duplicates,return=representation"),
            [{"source": source, "title": title}],
        )
        return rows[0]["id"]

    def upsert_chunks(self, rows: list[dict]) -> None:
        for i in range(0, len(rows), 100):
            http_json(
                "POST",
                f"{self.rest}/document_chunks?on_conflict=document_id,chunk_index",
                self._h("resolution=merge-duplicates,return=minimal"),
                rows[i : i + 100],
            )

    def delete_chunks_from(self, document_id: int, start_index: int) -> None:
        """청크 수가 줄어든 경우 남는 뒷부분을 지운다."""
        http_json(
            "DELETE",
            f"{self.rest}/document_chunks?document_id=eq.{document_id}&chunk_index=gte.{start_index}",
            self._h("return=minimal"),
        )

    def set_hash(self, document_id: int, content_hash: str) -> None:
        http_json(
            "PATCH",
            f"{self.rest}/documents?id=eq.{document_id}",
            self._h("return=minimal"),
            {"content_hash": content_hash, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        )

    def delete_document(self, source: str) -> None:
        q = urllib.parse.quote(source, safe="")
        http_json("DELETE", f"{self.rest}/documents?source=eq.{q}", self._h("return=minimal"))


def main() -> int:
    p = argparse.ArgumentParser(description="문서를 임베딩해 Supabase에 적재")
    p.add_argument("--dry-run", action="store_true", help="API 호출 없이 청크 통계만 출력")
    p.add_argument("--force", action="store_true", help="변경 여부와 상관없이 전부 다시 적재")
    args = p.parse_args()

    docs = []
    for rel in SOURCES:
        title, text = read_source(rel)
        chunks = chunk_markdown(text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        docs.append((rel, title, chunks, digest))

    if args.dry_run:
        total = 0
        for rel, title, chunks, _ in docs:
            lens = [len(c.content) for c in chunks]
            total += len(chunks)
            print(f"{rel:28s} 청크 {len(chunks):3d}개  최대 {max(lens, default=0):4d}자  [{title}]")
        print(f"합계 {total}개")
        return 0

    load_dotenv(ROOT / ".env")
    missing = [k for k in ("GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        print(f"환경 변수가 없습니다: {', '.join(missing)}", file=sys.stderr)
        return 1

    gemini = Gemini(os.environ["GEMINI_API_KEY"])
    db = Supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    existing = db.list_documents()

    for rel, title, chunks, digest in docs:
        if not args.force and existing.get(rel, {}).get("content_hash") == digest:
            print(f"건너뜀 (변경 없음): {rel}")
            continue
        doc_id = db.upsert_document(rel, title)
        vectors = gemini.embed([embed_text(title, c) for c in chunks])
        rows = [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "heading": c.heading,
                "content": c.content,
                "embedding": "[" + ",".join(f"{x:.7g}" for x in v) + "]",
            }
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        db.upsert_chunks(rows)
        db.delete_chunks_from(doc_id, len(rows))
        # 청크 적재가 끝난 뒤에 해시를 기록해야 중간 실패 시 다음 실행에서 다시 적재된다
        db.set_hash(doc_id, digest)
        print(f"적재 완료: {rel} ({len(rows)}개 청크)")

    for source in set(existing) - set(SOURCES):
        db.delete_document(source)
        print(f"삭제 (목록에서 빠짐): {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
