-- 청크를 섹터별 테이블로 나눈다 (Postgres 목록 파티션).
--
-- document_chunks는 이름은 그대로지만 실제 데이터는 섹터마다 따로 있는 파티션 테이블
-- (document_chunks_planning, document_chunks_hardware 등)에 들어간다. 벡터 인덱스도 섹터마다 따로 생긴다.
-- - 섹터를 지정해 검색하면 그 섹터 테이블만 읽는다. 데이터가 많아져도 읽는 양이 그 섹터 크기로 줄어든다.
-- - 섹터를 지정하지 않으면 모든 섹터 테이블에서 찾아 가까운 순으로 합친다 (지금 챗봇 동작).
-- - 적재·조회는 지금처럼 document_chunks 이름 하나로 한다. 어느 섹터 테이블에 들어갈지는 DB가 category로 정한다.
--
-- 기존 임베딩은 그대로 옮기므로 다시 임베딩하지 않는다. 한 트랜잭션으로 실행되므로 중간에 실패하면 전부 되돌아간다.

-- 1. 기존 청크를 임시로 옮겨 두고, 기존 테이블에 기대는 뷰·함수를 지운다
create temp table _old_chunks on commit drop as
select c.id, d.category, c.document_id, c.chunk_index, c.heading, c.content, c.embedding, c.created_at
from public.document_chunks c
join public.documents d on d.id = c.document_id;

drop view public.v_document_chunks;
drop view public.v_category_summary;
drop function public.match_document_chunks(extensions.vector, int, float, text);
drop table public.document_chunks;

-- 2. 청크의 섹터가 문서의 섹터와 항상 같도록 (문서 id, 섹터) 쌍으로 연결한다.
--    문서의 섹터를 바꾸면 청크도 따라 바뀌고, 해당 섹터 테이블로 자동으로 옮겨진다.
alter table public.documents add constraint documents_id_category_key unique (id, category);

-- 3. 섹터별로 나뉘는 청크 테이블
create sequence public.document_chunks_id_seq;

create table public.document_chunks (
  id bigint not null default nextval('public.document_chunks_id_seq'),
  category text not null,                 -- 섹터. 이 값으로 들어갈 파티션이 정해진다
  document_id bigint not null,
  chunk_index int not null,
  heading text not null default '',
  content text not null check (char_length(content) between 1 and 1500),
  embedding extensions.vector(768) not null,
  created_at timestamptz not null default now(),
  primary key (category, id),
  unique (category, document_id, chunk_index),
  foreign key (document_id, category)
    references public.documents (id, category) on delete cascade on update cascade
) partition by list (category);

alter sequence public.document_chunks_id_seq owned by public.document_chunks.id;

-- 섹터마다 테이블 하나. 새 섹터를 추가하면 여기처럼 파티션도 하나 만든다
-- (만들지 않으면 document_chunks_other에 들어간다).
create table public.document_chunks_planning partition of public.document_chunks for values in ('planning');
create table public.document_chunks_pipeline partition of public.document_chunks for values in ('pipeline');
create table public.document_chunks_model partition of public.document_chunks for values in ('model');
create table public.document_chunks_hardware partition of public.document_chunks for values in ('hardware');
create table public.document_chunks_field partition of public.document_chunks for values in ('field');
create table public.document_chunks_website partition of public.document_chunks for values in ('website');
create table public.document_chunks_other partition of public.document_chunks default;

-- 부모에 만든 인덱스는 섹터 테이블마다 따로 만들어진다 (섹터별 벡터 인덱스)
create index document_chunks_embedding_idx
  on public.document_chunks using hnsw (embedding extensions.vector_cosine_ops);
create index document_chunks_document_id_idx on public.document_chunks (document_id);

-- 4. 기존 청크를 옮기고, 개수가 맞는지 확인한다
insert into public.document_chunks (id, category, document_id, chunk_index, heading, content, embedding, created_at)
select id, category, document_id, chunk_index, heading, content, embedding, created_at
from _old_chunks;

select setval('public.document_chunks_id_seq', coalesce((select max(id) from public.document_chunks), 0) + 1, false);

do $check$
begin
  if (select count(*) from public.document_chunks) <> (select count(*) from _old_chunks) then
    raise exception '청크 개수가 맞지 않습니다. 마이그레이션을 되돌립니다';
  end if;
end
$check$;

-- 5. 공개 키로는 부모·섹터 테이블 모두 읽지 못하게 한다 (적재 스크립트와 Edge Function만 secret 키로 접근)
alter table public.document_chunks enable row level security;
alter table public.document_chunks_planning enable row level security;
alter table public.document_chunks_pipeline enable row level security;
alter table public.document_chunks_model enable row level security;
alter table public.document_chunks_hardware enable row level security;
alter table public.document_chunks_field enable row level security;
alter table public.document_chunks_website enable row level security;
alter table public.document_chunks_other enable row level security;

-- 6. 설명
comment on table public.document_chunks is '문서를 소제목 단위로 나눈 조각과 임베딩. 섹터(category)별 파티션 테이블로 나뉜다';
comment on column public.document_chunks.category is '섹터. 문서의 섹터와 같고, 이 값으로 섹터 테이블이 정해진다';
comment on column public.document_chunks.heading is '소제목 경로, 예) 2. 주요 파라미터 > 2.1 ...';
comment on column public.document_chunks.embedding is 'Gemini gemini-embedding-2, 768차원';
comment on table public.document_chunks_planning is '기획 섹터 청크';
comment on table public.document_chunks_pipeline is '엣지 파이프라인 섹터 청크';
comment on table public.document_chunks_model is '모델 섹터 청크';
comment on table public.document_chunks_hardware is '하드웨어 섹터 청크';
comment on table public.document_chunks_field is '현장 조사 섹터 청크';
comment on table public.document_chunks_website is '홈페이지 섹터 청크';
comment on table public.document_chunks_other is '파티션이 아직 없는 새 섹터의 청크 (새 섹터는 파티션을 만들어 옮긴다)';

-- 7. 뷰를 다시 만든다
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

create view public.v_document_chunks
with (security_invoker = true) as
select cat.name as category_name,
       c.category,
       d.source,
       d.title,
       c.chunk_index,
       c.heading,
       char_length(c.content) as length,
       c.content
from public.document_chunks c
join public.documents d on d.id = c.document_id
join public.document_categories cat on cat.slug = c.category
order by cat.sort_order, d.source, c.chunk_index;

comment on view public.v_category_summary is '섹터별 문서·청크 수 요약';
comment on view public.v_document_chunks is '섹터·문서 순으로 정렬한 청크 목록 (임베딩 제외)';

revoke all on public.v_category_summary, public.v_document_chunks from anon, authenticated;

-- 8. 검색 함수. 섹터를 지정하면 그 섹터 테이블만 읽는다 (파티션 프루닝).
--    지정하지 않으면 모든 섹터 테이블을 찾아 가까운 순으로 합친다. 반환 형식은 이전과 같다.
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
language plpgsql
stable
set search_path = public, extensions
as $fn$
declare
  n int := least(greatest(match_count, 1), 20);
begin
  if filter_category is null then
    return query
      select c.id, d.source, d.title, c.category, c.heading, c.content,
             1 - (c.embedding <=> query_embedding) as similarity
      from public.document_chunks c
      join public.documents d on d.id = c.document_id
      where 1 - (c.embedding <=> query_embedding) >= min_similarity
      order by c.embedding <=> query_embedding
      limit n;
  else
    return query
      select c.id, d.source, d.title, c.category, c.heading, c.content,
             1 - (c.embedding <=> query_embedding) as similarity
      from public.document_chunks c
      join public.documents d on d.id = c.document_id
      where c.category = filter_category
        and 1 - (c.embedding <=> query_embedding) >= min_similarity
      order by c.embedding <=> query_embedding
      limit n;
  end if;
end
$fn$;

revoke execute on function public.match_document_chunks(extensions.vector, int, float, text)
  from public, anon, authenticated;
