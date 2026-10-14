-- 문서를 섹터(분류)별로 관리한다.
-- 섹터마다 테이블을 나누지 않고 documents에 category 열을 둔다.
-- 청크는 한 테이블·한 벡터 인덱스에 그대로 두어야 질문 하나를 모든 섹터에서 한 번에 가까운 순으로 찾을 수 있다.
-- 섹터별로 보기 쉽게 뷰를 두고, 검색 함수는 필요할 때 섹터로 거를 수 있게 한다.

-- 섹터 목록. 새 섹터는 여기에 행을 추가하고 scripts/ingest_docs.py의 CATEGORIES에도 적는다.
create table public.document_categories (
  slug text primary key,                  -- 코드에서 쓰는 이름, 예) hardware
  name text not null,                     -- 화면에 보이는 이름, 예) 하드웨어
  description text not null default '',
  sort_order int not null default 0
);

insert into public.document_categories (slug, name, description, sort_order) values
  ('planning', '기획', '무엇을 누구를 위해 왜 만드는지 (PRD)', 1),
  ('pipeline', '엣지 파이프라인', '전처리·추적·카운팅·집계와 설정', 2),
  ('model', '모델', '비전·예측 모델 선정, 학습, 검증', 3),
  ('hardware', '하드웨어', '엣지 장비 부품, 전원, 배선', 4),
  ('field', '현장 조사', '설치 현장 조사 체크리스트', 5),
  ('website', '홈페이지', '제품 소개 페이지', 6);

alter table public.document_categories enable row level security;

-- 문서마다 섹터 하나. 기존 문서는 출처 경로로 채운 뒤 필수로 바꾼다.
alter table public.documents
  add column category text references public.document_categories(slug) on update cascade;

update public.documents set category = case source
  when 'PRD/PRD.md' then 'planning'
  when 'README.md' then 'pipeline'
  when 'report/preprocess.md' then 'pipeline'
  when 'report/model_plan.md' then 'model'
  when 'report/vision_models.md' then 'model'
  when 'docs.html' then 'model'
  when 'report/hardware.md' then 'hardware'
  when 'hardware.html' then 'hardware'
  when 'report/checklist.md' then 'field'
  when 'index.html' then 'website'
end;

alter table public.documents alter column category set not null;
create index documents_category_idx on public.documents (category);

-- 대시보드에서 읽기 쉽도록 테이블·열 설명을 단다.
comment on table public.document_categories is '챗봇 검색 문서의 섹터(분류) 목록';
comment on table public.documents is '챗봇 검색 대상 원본 문서. 1행 = 저장소 파일 1개';
comment on column public.documents.source is '저장소 기준 경로, 예) report/preprocess.md';
comment on column public.documents.category is '섹터 (document_categories.slug)';
comment on column public.documents.content_hash is '원문 sha256. 청크 적재가 끝난 뒤에만 채워 바뀐 문서만 다시 임베딩한다';
comment on table public.document_chunks is '문서를 소제목 단위로 나눈 조각과 임베딩. 모든 섹터가 벡터 인덱스 하나를 같이 쓴다';
comment on column public.document_chunks.heading is '소제목 경로, 예) 2. 주요 파라미터 > 2.1 ...';
comment on column public.document_chunks.embedding is 'Gemini gemini-embedding-2, 768차원';

-- 섹터별 문서·청크 수 요약
create view public.v_category_summary
with (security_invoker = true) as
select cat.slug as category,
       cat.name as category_name,
       count(distinct d.id) as documents,
       count(c.id) as chunks,
       max(d.updated_at) as last_updated
from public.document_categories cat
left join public.documents d on d.category = cat.slug
left join public.document_chunks c on c.document_id = d.id
group by cat.slug, cat.name, cat.sort_order
order by cat.sort_order;

-- 섹터·문서 순으로 정렬한 청크 목록 (임베딩 벡터는 빼고 읽기 쉬운 열만)
create view public.v_document_chunks
with (security_invoker = true) as
select cat.name as category_name,
       d.category,
       d.source,
       d.title,
       c.chunk_index,
       c.heading,
       char_length(c.content) as length,
       c.content
from public.document_chunks c
join public.documents d on d.id = c.document_id
join public.document_categories cat on cat.slug = d.category
order by cat.sort_order, d.source, c.chunk_index;

comment on view public.v_category_summary is '섹터별 문서·청크 수 요약';
comment on view public.v_document_chunks is '섹터·문서 순으로 정렬한 청크 목록 (임베딩 제외)';

-- 뷰도 공개 키로는 읽지 못하게 한다 (security_invoker라 RLS가 적용되지만 권한도 막아 둔다)
revoke all on public.v_category_summary, public.v_document_chunks from anon, authenticated;

-- 검색 함수에 섹터 필터와 섹터 결과 열을 추가한다. 반환 형식이 바뀌므로 지우고 다시 만든다.
-- filter_category가 null(기본값)이면 지금처럼 전체 섹터에서 찾으므로 기존 호출은 그대로 동작한다.
drop function public.match_document_chunks(extensions.vector, int, float);

create function public.match_document_chunks(
  query_embedding extensions.vector(768),
  match_count int default 6,
  min_similarity float default 0.3,
  filter_category text default null
)
returns table (
  id bigint,
  source text,
  title text,
  category text,
  heading text,
  content text,
  similarity float
)
language sql
stable
set search_path = public, extensions
as $$
  select c.id, d.source, d.title, d.category, c.heading, c.content,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.document_chunks c
  join public.documents d on d.id = c.document_id
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
    and (filter_category is null or d.category = filter_category)
  order by c.embedding <=> query_embedding
  limit least(greatest(match_count, 1), 20);
$$;

revoke execute on function public.match_document_chunks(extensions.vector, int, float, text)
  from public, anon, authenticated;
