"""챗봇 적재 문서의 섹터 설정이 DB 마이그레이션과 맞는지 확인한다 (API 호출 없음)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_docs  # noqa: E402

MIGRATION = ROOT / "supabase" / "migrations" / "20261014000000_rag_document_categories.sql"
PARTITION_MIGRATION = ROOT / "supabase" / "migrations" / "20261015000000_rag_partition_chunks_by_category.sql"


def test_모든_적재_문서에_섹터가_있다():
    missing = [rel for rel in ingest_docs.SOURCES if ingest_docs.CATEGORIES.get(rel) not in ingest_docs.CATEGORY_SLUGS]
    assert missing == []


def test_섹터_목록이_마이그레이션과_같다():
    sql = MIGRATION.read_text(encoding="utf-8")
    inserted = re.findall(r"\('([a-z]+)', '[^']+', '[^']*', \d+\)", sql)
    assert tuple(inserted) == ingest_docs.CATEGORY_SLUGS


def test_마이그레이션의_기존_문서_분류가_스크립트와_같다():
    sql = MIGRATION.read_text(encoding="utf-8")
    backfill = dict(re.findall(r"when '([^']+)' then '([a-z]+)'", sql))
    # 마이그레이션은 그때 있던 문서만 분류한다. 나중에 추가한 문서는 스크립트가 섹터를 넣는다.
    assert backfill
    assert {src: ingest_docs.CATEGORIES.get(src) for src in backfill} == backfill


def test_섹터마다_청크_파티션_테이블이_있다():
    sql = PARTITION_MIGRATION.read_text(encoding="utf-8")
    partitions = re.findall(
        r"create table public\.document_chunks_([a-z]+) partition of public\.document_chunks for values in \('([a-z]+)'\)",
        sql,
    )
    assert all(name == value for name, value in partitions)  # 테이블 이름과 섹터 값이 같아야 찾기 쉽다
    assert tuple(value for _, value in partitions) == ingest_docs.CATEGORY_SLUGS
    # 섹터 테이블마다 공개 키 접근을 막는다
    for slug in ingest_docs.CATEGORY_SLUGS:
        assert f"alter table public.document_chunks_{slug} enable row level security;" in sql
