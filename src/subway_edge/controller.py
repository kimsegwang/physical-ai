"""시간대(혼잡도)·날씨별 설정 자동 전환 (가이드 5장).

- 혼잡도: 직전 1분 집계의 게이트당 인원(입장+퇴장)으로 peak / normal / quiet 전환.
  전환 기준과 복귀 기준을 다르게 둬서(히스테리시스) 경계에서 설정이 왔다 갔다 하지 않게 한다
- 날씨: 클라우드가 날씨 API를 보고 set_weather()로 알려준다
- 최종 설정 = 기본 설정 ← 혼잡도 프로필 ← 날씨 프로필 ← 외투 프로필 순으로 덮어쓴다
"""

from __future__ import annotations

from .config import WEATHER_CONDITIONS, PipelineConfig, apply_overrides

MODES = ("peak", "normal", "quiet")


class AdaptiveController:
    def __init__(self, base: PipelineConfig) -> None:
        self.base = base
        self.mode = "normal"
        self.weather = "clear"
        self.heavy_coat = False
        self._config = self._build()

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def set_weather(self, condition: str, heavy_coat: bool = False) -> bool:
        """날씨를 바꾸고 설정이 바뀌었으면 True."""
        if condition not in WEATHER_CONDITIONS:
            raise ValueError(f"날씨는 {WEATHER_CONDITIONS} 중 하나여야 합니다: {condition}")
        if (condition, heavy_coat) == (self.weather, self.heavy_coat):
            return False
        self.weather, self.heavy_coat = condition, heavy_coat
        self._config = self._build()
        return True

    def on_minute(self, record: dict, num_gates: int) -> bool:
        """1분 집계를 받아 혼잡도 모드를 갱신하고, 모드가 바뀌었으면 True."""
        a = self.base.adaptive
        if not a.enabled:
            return False
        rate = (record["in"] + record["out"]) / max(1, num_gates)
        mode = self.mode
        if mode == "peak":
            if rate < a.peak_exit_per_gate:
                mode = "quiet" if rate <= a.quiet_enter_per_gate else "normal"
        elif mode == "quiet":
            if rate >= a.quiet_exit_per_gate:
                mode = "peak" if rate >= a.peak_enter_per_gate else "normal"
        else:
            if rate >= a.peak_enter_per_gate:
                mode = "peak"
            elif rate <= a.quiet_enter_per_gate:
                mode = "quiet"
        if mode == self.mode:
            return False
        self.mode = mode
        self._config = self._build()
        return True

    def _build(self) -> PipelineConfig:
        a = self.base.adaptive
        cfg = self.base
        if not a.enabled:
            return cfg
        cfg = apply_overrides(cfg, a.mode_profiles.get(self.mode, {}))
        cfg = apply_overrides(cfg, a.weather_profiles.get(self.weather, {}))
        if self.heavy_coat:
            cfg = apply_overrides(cfg, a.heavy_coat_profile)
        return cfg
