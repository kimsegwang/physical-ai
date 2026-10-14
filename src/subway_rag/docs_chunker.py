"""RAG 챗봇용 문서 청킹.

마크다운·HTML 문서를 소제목 단위로 나누고, 긴 구간은 문단 → 줄 → 글자 순으로 잘라
한 조각이 max_chars를 넘지 않게 만든다. 표는 잘라도 머리행을 조각마다 다시 붙인다.
DB의 document_chunks.content 길이 제한(1500자)보다 넉넉히 작게 유지한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

MAX_CHARS = 900
OVERLAP_CHARS = 150

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?。])\s+")


@dataclass(frozen=True)
class Chunk:
    """문서 조각 하나."""

    heading: str  # 소제목 경로, 예) "2. 기준별 분석 > 2.1 속도"
    content: str


def chunk_markdown(
    text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS
) -> list[Chunk]:
    """마크다운을 소제목 구간별로 나눈 뒤 길이 제한에 맞춰 조각낸다."""
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        for piece in _pack_blocks(_split_blocks(body), max_chars, overlap):
            chunks.append(Chunk(heading=heading, content=piece))
    return chunks


def document_title(text: str, fallback: str) -> str:
    """첫 번째 '# ' 제목을 문서 제목으로 쓴다. 없으면 fallback."""
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) == 1 and m.group(2):
            return m.group(2)
    return fallback


def html_to_markdown(html: str) -> str:
    """HTML 본문 텍스트를 마크다운 비슷한 형태로 뽑는다 (제목은 #, 블록은 줄바꿈)."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in "".join(parser.out).splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# 내부 구현


def _split_sections(text: str) -> list[tuple[str, str]]:
    """제목 줄을 기준으로 (소제목 경로, 본문) 목록을 만든다. 코드 블록 안의 #은 무시한다."""
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in path), body))
        buf.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING_RE.match(line)
        if m and m.group(2):
            flush()
            level = len(m.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()
    return sections


def _split_blocks(body: str) -> list[str]:
    """빈 줄 기준으로 문단(블록)을 나눈다. 코드 블록은 통째로 하나의 블록으로 둔다."""
    blocks: list[str] = []
    cur: list[str] = []
    in_code = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def _pack_blocks(blocks: list[str], max_chars: int, overlap: int) -> list[str]:
    """블록을 max_chars 안에서 최대한 붙인다. 넘치는 블록은 더 잘게 쪼갠다.

    새 조각을 시작할 때 직전 블록이 overlap 이하로 짧으면 앞에 다시 붙여 문맥을 잇는다.
    """
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(_split_long(block, max_chars, overlap))

    out: list[str] = []
    cur = ""
    last = ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur = f"{cur}\n\n{p}"
        else:
            out.append(cur)
            carry = last if last and len(last) <= overlap and len(last) + 2 + len(p) <= max_chars else ""
            cur = f"{carry}\n\n{p}" if carry else p
        last = p
    if cur:
        out.append(cur)
    return out


def _split_long(block: str, max_chars: int, overlap: int) -> list[str]:
    """한 블록이 max_chars를 넘으면 줄 → 문장 → 글자 순으로 나눈다."""
    if len(block) <= max_chars:
        return [block]

    lines = block.splitlines()
    if _is_table(lines):
        return _split_table(lines, max_chars, overlap)
    if len(lines) > 1:
        units = lines
        sep = "\n"
    else:
        units = [s for s in _SENTENCE_RE.split(block) if s and s.strip()]
        sep = " "
        if len(units) <= 1:
            return _hard_split(block, max_chars, overlap)

    out: list[str] = []
    cur = ""
    for u in units:
        if len(u) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_hard_split(u, max_chars, overlap))
        elif not cur:
            cur = u
        elif len(cur) + len(sep) + len(u) <= max_chars:
            cur = f"{cur}{sep}{u}"
        else:
            out.append(cur)
            cur = u
    if cur:
        out.append(cur)
    return out


def _is_table(lines: list[str]) -> bool:
    return len(lines) >= 2 and all(ln.lstrip().startswith("|") for ln in lines[:2])


def _split_table(lines: list[str], max_chars: int, overlap: int) -> list[str]:
    """마크다운 표를 행 단위로 나누고, 조각마다 머리행(+구분행)을 다시 붙인다."""
    has_sep = len(lines) > 1 and re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", lines[1]) is not None
    header = lines[:2] if has_sep else lines[:1]
    rows = lines[len(header):]
    head = "\n".join(header)
    if len(head) > max_chars // 2:
        return _hard_split("\n".join(lines), max_chars, overlap)

    out: list[str] = []
    cur: list[str] = []
    size = len(head)
    for row in rows:
        if len(row) + len(head) + 1 > max_chars:
            row = row[: max_chars - len(head) - 2] + "…"
        if cur and size + 1 + len(row) > max_chars:
            out.append("\n".join([head, *cur]))
            cur, size = [], len(head)
        cur.append(row)
        size += 1 + len(row)
    if cur:
        out.append("\n".join([head, *cur]))
    return out


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """마지막 수단: 글자 수로 자르되 앞 조각 끝 overlap 글자를 겹친다."""
    step = max(1, max_chars - overlap)
    out = []
    for start in range(0, len(text), step):
        out.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
    return out


class _TextExtractor(HTMLParser):
    """script·style·svg를 버리고 본문 텍스트만 모은다."""

    # nav는 메뉴 이름뿐이라 검색에 도움이 안 된다
    _SKIP = {"script", "style", "svg", "noscript", "head", "nav"}
    _BLOCK = {
        "p", "div", "section", "li", "ul", "ol", "tr", "table", "br",
        "header", "footer", "main", "article", "aside",
    }
    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip_depth = 0
        self._icon_depth = 0  # Material 아이콘 <span>은 글리프 이름(arrow_forward 등)이라 버린다
        self._in_heading = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "span" and (self._icon_depth or "material-symbols" in (dict(attrs).get("class") or "")):
            self._icon_depth += 1
        elif self._icon_depth:
            return
        elif tag in self._SKIP:
            self._skip_depth += 1
        elif self._skip_depth:
            return
        elif tag in self._HEADINGS:
            self._in_heading = True
            self.out.append("\n\n" + "#" * self._HEADINGS[tag] + " ")
        elif self._in_heading:
            # 제목 안의 <br>·<span>은 줄을 바꾸지 않고 한 줄 제목으로 유지
            if tag == "br":
                self.out.append(" ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._icon_depth:
            if tag == "span":
                self._icon_depth -= 1
            return
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth:
            return
        elif tag in self._HEADINGS:
            self._in_heading = False
            self.out.append("\n\n")
        elif self._in_heading:
            return
        elif tag in ("p", "section", "table", "ul", "ol"):
            self.out.append("\n\n")
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and not self._icon_depth:
            self.out.append(data.replace("\n", " "))
