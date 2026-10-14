"""챗봇 적재 문서의 섹터 설정이 DB 마이그레이션과 맞는지 확인한다 (API 호출 없음)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_docs  # noqa: E402

MIGRATION = ROOT / "supabase" / "migrations" / "20261014000000_rag_document_categories.sql"


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
    assert backfill == ingest_docs.CATEGORIES
