import pytest

from fdtdx.core.switch import OnOffSwitch, is_on_at_time_step, is_on_at_time_step_from_switch

# ── is_on_at_time_step (standalone function) ──────────────────────────────


class TestIsOnAtTimeStep:
    """Tests for the standalone is_on_at_time_step function."""

    def _defaults(self, **overrides):
        """Return default kwargs for is_on_at_time_step with overrides."""
        base = dict(
            is_always_off=False,
            start_time=None,
            start_after_periods=None,
            end_time=None,
            end_after_periods=None,
            on_for_time=None,
            on_for_periods=None,
            time_step=0,
            time_step_duration=1.0,
            period=None,
        )
        base.update(overrides)
        return base

    # --- always off ---
    def test_always_off(self):
        assert is_on_at_time_step(**self._defaults(is_always_off=True)) is False

    # --- default (no start/end) → always on ---
    def test_default_always_on(self):
        assert is_on_at_time_step(**self._defaults()) is True

    # --- start_time / end_time boundaries ---
    def test_start_time_before(self):
        assert is_on_at_time_step(**self._defaults(start_time=5.0, time_step=3)) is False

    def test_start_time_exact(self):
        assert is_on_at_time_step(**self._defaults(start_time=5.0, time_step=5)) is True

    def test_end_time_exact(self):
        assert is_on_at_time_step(**self._defaults(end_time=5.0, time_step=5)) is True

    def test_end_time_after(self):
        assert is_on_at_time_step(**self._defaults(end_time=5.0, time_step=7)) is False

    # --- start_after_periods / end_after_periods ---
    def test_start_after_periods(self):
        kw = self._defaults(start_after_periods=2.0, period=10.0)
        assert is_on_at_time_step(**{**kw, "time_step": 19}) is False
        assert is_on_at_time_step(**{**kw, "time_step": 20}) is True

    def test_end_after_periods(self):
        kw = self._defaults(end_after_periods=3.0, period=10.0)
        assert is_on_at_time_step(**{**kw, "time_step": 30}) is True
        assert is_on_at_time_step(**{**kw, "time_step": 31}) is False

    # --- on_for_time derives start from end ---
    def test_on_for_time_with_end_time(self):
        # end=10, on_for_time=4 → start=6
        kw = self._defaults(on_for_time=4.0, end_time=10.0)
        assert is_on_at_time_step(**{**kw, "time_step": 5}) is False
        assert is_on_at_time_step(**{**kw, "time_step": 6}) is True

    # --- on_for_time derives end from start ---
    def test_on_for_time_with_start_time(self):
        # start=5, on_for_time=3 → end=8
        kw = self._defaults(start_time=5.0, on_for_time=3.0)
        assert is_on_at_time_step(**{**kw, "time_step": 8}) is True
        assert is_on_at_time_step(**{**kw, "time_step": 9}) is False

    # --- on_for_periods converts to on_for_time ---
    def test_on_for_periods_with_end_time(self):
        # end=20, on_for_periods=1, period=10 → on_for_time=10 → start=10
        kw = self._defaults(on_for_periods=1.0, end_time=20.0, period=10.0)
        assert is_on_at_time_step(**{**kw, "time_step": 9}) is False
        assert is_on_at_time_step(**{**kw, "time_step": 10}) is True

    def test_on_for_periods_with_start_time(self):
        # start=5, on_for_periods=2, period=10 → on_for_time=20 → end=25
        kw = self._defaults(start_time=5.0, on_for_periods=2.0, period=10.0)
        assert is_on_at_time_step(**{**kw, "time_step": 25}) is True
        assert is_on_at_time_step(**{**kw, "time_step": 26}) is False

    # --- time_step_duration scaling ---
    def test_time_step_duration_scales(self):
        kw = self._defaults(start_time=10.0, time_step_duration=0.5)
        assert is_on_at_time_step(**{**kw, "time_step": 19}) is False
        assert is_on_at_time_step(**{**kw, "time_step": 20}) is True

    # --- errors ---
    def test_error_need_period(self):
        with pytest.raises(Exception, match="Need to specify period"):
            is_on_at_time_step(**self._defaults(start_after_periods=1.0))

    def test_error_conflicting_start_specs(self):
        with pytest.raises(Exception, match="Invalid start time"):
            is_on_at_time_step(
                **self._defaults(
                    start_time=0.0,
                    start_after_periods=1.0,
                    period=10.0,
                )
            )

    def test_error_conflicting_end_specs(self):
        with pytest.raises(Exception, match="Invalid end time"):
            is_on_at_time_step(
                **self._defaults(
                    end_time=10.0,
                    end_after_periods=2.0,
                    period=5.0,
                )
            )


# ── OnOffSwitch class ─────────────────────────────────────────────────────


class TestOnOffSwitch:
    """Tests for OnOffSwitch class methods."""

    def test_calculate_on_list_default(self):
        switch = OnOffSwitch()
        result = switch.calculate_on_list(num_total_time_steps=5, time_step_duration=1.0)
        assert result == [True, True, True, True, True]

    def test_calculate_on_list_with_interval(self):
        switch = OnOffSwitch(interval=3)
        result = switch.calculate_on_list(num_total_time_steps=7, time_step_duration=1.0)
        assert result == [True, False, False, True, False, False, True]

    def test_calculate_on_list_fixed_time_steps(self):
        switch = OnOffSwitch(fixed_on_time_steps=[1, 3, 4])
        result = switch.calculate_on_list(num_total_time_steps=6, time_step_duration=1.0)
        assert result == [False, True, False, True, True, False]

    def test_time_step_to_on_arr_idx_maps_on_steps(self):
        switch = OnOffSwitch(start_time=1.0, end_time=3.0)
        result = switch.calculate_time_step_to_on_arr_idx(
            num_total_time_steps=6,
            time_step_duration=1.0,
        )
        assert result == [-1, 0, 1, 2, -1, -1]


# ── is_on_at_time_step_from_switch ────────────────────────────────────────


class TestIsOnAtTimeStepFromSwitch:
    """Tests for the is_on_at_time_step_from_switch helper."""

    def test_delegates_to_switch(self):
        switch = OnOffSwitch(start_time=5.0, end_time=10.0)
        assert is_on_at_time_step_from_switch(time_step=3, time_step_duration=1.0, switch=switch) is False
        assert is_on_at_time_step_from_switch(time_step=5, time_step_duration=1.0, switch=switch) is True
