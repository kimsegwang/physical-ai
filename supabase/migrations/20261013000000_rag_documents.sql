-- RAG 검색 챗봇용 문서·청크 저장소
-- 임베딩: Gemini gemini-embedding-2, 768차원 (output_dimensionality=768)

create extension if not exists vector with schema extensions;

-- 원본 문서 1개 = 1행. content_hash로 바뀐 문서만 다시 임베딩한다.
create table public.documents (
  id bigint generated always as identity primary key,
  source text not null unique,            -- 저장소 기준 경로, 예) report/preprocess.md
  title text not null,
  content_hash text not null default '',  -- 원문 sha256. 청크 적재가 끝난 뒤에만 채운다
  updated_at timestamptz not null default now()
);

-- 문서를 잘게 나눈 조각. 너무 긴 조각은 검색 품질과 토큰 비용을 떨어뜨리므로 길이를 제한한다.
create table public.document_chunks (
  id bigint generated always as identity primary key,
  document_id bigint not null references public.documents(id) on delete cascade,
  chunk_index int not null,
  heading text not null default '',       -- 소제목 경로, 예) 2. 기준별 분석 > 2.1 속도
  content text not null check (char_length(content) between 1 and 1500),
  embedding extensions.vector(768) not null,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index document_chunks_embedding_idx
  on public.document_chunks using hnsw (embedding extensions.vector_cosine_ops);

-- RLS를 켜고 정책은 두지 않는다 → anon/publishable 키로는 직접 읽기·쓰기 불가.
-- 적재 스크립트와 Edge Function만 secret 키로 접근한다.
alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;

-- 질문 임베딩과 가까운 청크를 코사인 유사도 순으로 돌려준다.
create or replace function public.match_document_chunks(
  query_embedding extensions.vector(768),
  match_count int default 6,
  min_similarity float default 0.3
)
returns table (
  id bigint,
  source text,
  title text,
  heading text,
  content text,
  similarity float
)
language sql
stable
set search_path = public, extensions
as $$
  select c.id, d.source, d.title, c.heading, c.content,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.document_chunks c
  join public.documents d on d.id = c.document_id
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by c.embedding <=> query_embedding
  limit least(greatest(match_count, 1), 20);
$$;

revoke execute on function public.match_document_chunks(extensions.vector, int, float)
  from public, anon, authenticated;
