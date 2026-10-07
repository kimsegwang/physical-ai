"""8단계: 이중 통과선 카운팅.

- 선 A와 선 B를 순서대로 지나야 1명으로 센다 (A→B = a_to_b 방향, B→A = 반대 방향)
- 첫 선을 넘었다가 같은 선으로 되돌아가면 무효
- 같은 ID는 방향별로 1회만 세고, 카운트 후 recount_block_sec 동안은 어떤 방향도 다시 세지 않는다
- 궤적이 min_track_len보다 짧으면 통과를 보류했다가 길이가 차면 센다 (그 전에 사라지면 버림)
- 개찰구 단위 쿨다운은 쓰지 않는다 (꼬리물기 때 두 번째 사람이 누락됨)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import CountingConfig, GateConfig, Point
from .tracking import Track


@dataclass(frozen=True)
class CountEvent:
    ts: float
    track_id: int
    gate: str
    direction: str  # "in" 또는 "out"


@dataclass
class _TrackState:
    last_anchor: tuple[float, float]
    armed: dict[str, str] = field(default_factory=dict)  # 게이트별로 먼저 넘은 선 ("a"/"b")
    pending: list[tuple[str, str]] = field(default_factory=list)  # 길이 미달로 보류된 (게이트, 방향)
    counted: set[str] = field(default_factory=set)  # 이미 센 방향
    last_count_ts: float | None = None


def _side(p: tuple[float, float], a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing_param(p0: tuple[float, float], p1: tuple[float, float], line: list[Point]) -> float | None:
    """p0→p1 이동이 선분을 지나면 이동 경로상의 위치(0~1)를, 아니면 None을 돌려준다."""
    a, b = line
    s0, s1 = _side(p0, a, b), _side(p1, a, b)
    if (s0 >= 0) == (s1 >= 0):
        return None
    # 선분의 양 끝이 이동 경로의 서로 다른 쪽에 있어야 선분 범위 안에서 교차
    d0, d1 = _side(tuple(a), p0, p1), _side(tuple(b), p0, p1)
    if d0 * d1 > 0:
        return None
    return s0 / (s0 - s1)


class LineCounter:
    def __init__(self, cfg: CountingConfig) -> None:
        self._states: dict[int, _TrackState] = {}
        self.totals = {"in": 0, "out": 0}
        self.apply_config(cfg)

    def apply_config(self, cfg: CountingConfig) -> None:
        self.cfg = cfg

    def _direction(self, first: str) -> str:
        if first == "a":
            return self.cfg.a_to_b
        return "in" if self.cfg.a_to_b == "out" else "out"

    def update(self, tracks: list[Track], ts: float, alive_ids: set[int]) -> list[CountEvent]:
        """이번 프레임에 검출과 매칭된 트랙으로 통과를 판정한다.

        놓친 동안의 예측 위치로는 판정하지 않고, 다시 검출됐을 때 마지막 관측 위치와 이어서 본다.
        """
        cfg = self.cfg
        events: list[CountEvent] = []
        for track in tracks:
            anchor = track.anchor(cfg.anchor)
            state = self._states.get(track.track_id)
            if state is None:
                self._states[track.track_id] = _TrackState(anchor)
                continue
            for gate in cfg.gates:
                self._check_gate(state, gate, state.last_anchor, anchor)
            state.last_anchor = anchor
            if state.pending and track.hits >= cfg.min_track_len:
                for gate_name, direction in state.pending:
                    event = self._try_count(state, track.track_id, gate_name, direction, ts)
                    if event:
                        events.append(event)
                state.pending.clear()

        for tid in list(self._states):
            if tid not in alive_ids:
                del self._states[tid]
        return events

    def _check_gate(self, state: _TrackState, gate: GateConfig, p0, p1) -> None:
        crossings = []
        for name, line in (("a", gate.line_a), ("b", gate.line_b)):
            t = crossing_param(p0, p1, line)
            if t is not None:
                crossings.append((t, name))
        # 한 프레임에 두 선을 모두 지난 경우 이동 순서대로 처리
        for _, name in sorted(crossings):
            first = state.armed.get(gate.name)
            if first is None:
                state.armed[gate.name] = name
            elif first == name:
                state.armed.pop(gate.name)  # 같은 선으로 되돌아감 → 무효
            else:
                state.armed.pop(gate.name)
                state.pending.append((gate.name, self._direction(first)))

    def _try_count(self, state: _TrackState, track_id: int, gate: str, direction: str, ts: float) -> CountEvent | None:
        if direction in state.counted:
            return None
        if state.last_count_ts is not None and ts - state.last_count_ts < self.cfg.recount_block_sec:
            return None
        state.counted.add(direction)
        state.last_count_ts = ts
        self.totals[direction] += 1
        return CountEvent(ts, track_id, gate, direction)
