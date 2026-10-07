"""집계 결과 전송.

전송 방식은 Publisher 인터페이스로 분리한다. 지금은 표준 출력(JSON 한 줄)으로 내보내고,
나중에 MQTT 등으로 바꿀 때는 같은 인터페이스를 구현한 클래스만 넘기면 된다.
보내는 것은 1분 집계 레코드(숫자)뿐이며, 영상·프레임은 보내지 않는다.
"""

from __future__ import annotations

import json
import sys
from typing import Protocol, TextIO


class Publisher(Protocol):
    def publish(self, record: dict) -> None:
        """1분 집계 레코드 하나를 전송한다."""
        ...

    def close(self) -> None:
        """남은 전송을 마무리하고 연결을 닫는다."""
        ...


class StdoutPublisher:
    """레코드를 JSON 한 줄씩 출력한다."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def publish(self, record: dict) -> None:
        self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.flush()
