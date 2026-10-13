"""RAG 문서 청킹 테스트."""

from __future__ import annotations

from subway_edge.docs_chunker import (
    chunk_markdown,
    document_title,
    html_to_markdown,
)


def test_소제목_경로가_청크에_붙는다():
    md = "# 문서\n\n## 1. 결론\n\n엣지를 선택한다.\n\n### 1.1 속도\n\n1초 안에 응답한다.\n\n## 2. 비용\n\n싸다."
    chunks = chunk_markdown(md)
    assert [(c.heading, c.content) for c in chunks] == [
        ("문서 > 1. 결론", "엣지를 선택한다."),
        ("문서 > 1. 결론 > 1.1 속도", "1초 안에 응답한다."),
        ("문서 > 2. 비용", "싸다."),
    ]


def test_모든_청크는_최대_길이를_넘지_않는다():
    para = "가나다라마바사아자차카타파하. " * 40  # 문장 하나짜리 긴 문단
    long_line = "x" * 2500  # 공백·문장부호 없는 아주 긴 줄
    md = f"# 제목\n\n{para}\n\n{long_line}\n\n짧은 문단"
    chunks = chunk_markdown(md, max_chars=300, overlap=50)
    assert chunks
    assert all(len(c.content) <= 300 for c in chunks)
    # 잘린 내용이 사라지지 않는다
    joined = "".join(c.content for c in chunks)
    assert "짧은 문단" in joined
    assert joined.count("x") >= 2500


def test_긴_표는_조각마다_머리행을_다시_붙인다():
    header = "| 구분 | 값 |\n|---|---|"
    rows = "\n".join(f"| 항목{i} | {'값' * 20} |" for i in range(40))
    chunks = chunk_markdown(f"# 표\n\n{header}\n{rows}", max_chars=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.content.startswith(header)
        assert len(c.content) <= 300
    body = "\n".join(c.content for c in chunks)
    assert all(f"항목{i} |" in body for i in range(40))


def test_코드_블록_안의_샵은_제목이_아니다():
    md = "# 문서\n\n```bash\n# 주석입니다\npython run.py\n```\n"
    chunks = chunk_markdown(md)
    assert len(chunks) == 1
    assert chunks[0].heading == "문서"
    assert "# 주석입니다" in chunks[0].content


def test_짧은_문단은_한_청크로_묶인다():
    md = "# 문서\n\n" + "\n\n".join(f"문단 {i}" for i in range(5))
    chunks = chunk_markdown(md, max_chars=900)
    assert len(chunks) == 1


def test_문서_제목():
    assert document_title("앞말\n# 진짜 제목\n## 소제목", "기본") == "진짜 제목"
    assert document_title("## 소제목만", "기본") == "기본"


def test_HTML_본문만_뽑고_아이콘과_스크립트는_버린다():
    html = """
    <html><head><title>T</title><style>.a{}</style></head><body>
    <nav><a href="#">메뉴</a></nav>
    <h1>역에 가기 전에,<br/><span>혼잡을 먼저</span> 확인하세요</h1>
    <p>개찰구 <span class="material-symbols-outlined">arrow_forward</span>카메라</p>
    <script>var x = 1;</script>
    <svg><text>그림</text></svg>
    </body></html>
    """
    md = html_to_markdown(html)
    assert "# 역에 가기 전에, 혼잡을 먼저 확인하세요" in md
    assert "개찰구 카메라" in md
    for junk in ("arrow_forward", "var x", "그림", "메뉴", ".a{}"):
        assert junk not in md
