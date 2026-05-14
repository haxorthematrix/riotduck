"""Tests for the EventTracker re-observation / dedup logic."""
from __future__ import annotations

from riotduck.dedup import EventTracker
from riotduck.events import Detection


def _det(type_: str, ts: float, center_hz: float = 433.92e6,
         bw_hz: float = 10e3, range_name: str = "ism_433") -> Detection:
    return Detection.new(
        type=type_,
        ts=ts,
        range_name=range_name,
        device_serial="dev-0",
        center_hz=center_hz,
        bw_hz=bw_hz,
        power_dbfs=-40.0,
        snr_db=30.0,
        bins=[100],
        first_seen_ts=ts,
        last_seen_ts=ts,
    )


def test_first_appearance_is_published():
    tr = EventTracker(re_observation_s=30.0)
    d = _det("appearance", ts=100.0)
    out = tr.observe(d)
    assert out is not None
    assert out.id == d.id
    assert len(tr.active_events()) == 1


def test_repeat_appearance_within_window_is_suppressed():
    tr = EventTracker(re_observation_s=30.0)
    first = _det("appearance", ts=100.0)
    second = _det("appearance", ts=110.0)        # same freq, 10 s later
    assert tr.observe(first) is not None
    assert tr.observe(second) is None
    # the tracked event's last_seen should be refreshed
    evts = tr.active_events()
    assert len(evts) == 1
    assert evts[0].last_seen_ts == 110.0


def test_appearance_outside_window_is_a_new_event():
    tr = EventTracker(re_observation_s=30.0)
    first = _det("appearance", ts=100.0)
    third = _det("appearance", ts=200.0)         # 100 s later — fresh event
    out1 = tr.observe(first)
    out2 = tr.observe(third)
    assert out1 is not None and out2 is not None
    assert out1.id != out2.id


def test_disappearance_after_appearance_publishes_with_same_id():
    tr = EventTracker(re_observation_s=30.0)
    app = _det("appearance", ts=100.0)
    disap = _det("disappearance", ts=110.0)
    out_app = tr.observe(app)
    out_disap = tr.observe(disap)
    assert out_disap is not None
    assert out_disap.id == out_app.id
    # tracked event is now silent
    all_evts = tr.all_events()
    assert len(all_evts) == 1
    assert all_evts[0].state == "silent"


def test_reappearance_after_disappearance_publishes_with_original_id():
    tr = EventTracker(re_observation_s=30.0)
    app = _det("appearance", ts=100.0)
    disap = _det("disappearance", ts=110.0)
    again = _det("appearance", ts=115.0)         # back within window
    out_app = tr.observe(app)
    tr.observe(disap)
    out_again = tr.observe(again)
    assert out_again is not None                 # NOT suppressed
    assert out_again.id == out_app.id            # but reuses original id


def test_frequencies_outside_tolerance_are_separate():
    tr = EventTracker(re_observation_s=30.0, min_freq_tolerance_hz=5000.0)
    a = _det("appearance", ts=100.0, center_hz=433.92e6)
    b = _det("appearance", ts=100.5, center_hz=433.93e6)   # 10 kHz away
    out_a = tr.observe(a)
    out_b = tr.observe(b)
    assert out_a is not None and out_b is not None
    assert out_a.id != out_b.id


def test_frequencies_inside_tolerance_collapse():
    tr = EventTracker(re_observation_s=30.0, min_freq_tolerance_hz=5000.0)
    a = _det("appearance", ts=100.0, center_hz=433.920e6)
    b = _det("appearance", ts=100.5, center_hz=433.921e6)   # 1 kHz away
    assert tr.observe(a) is not None
    assert tr.observe(b) is None


def test_wider_bandwidth_widens_tolerance():
    tr = EventTracker(re_observation_s=30.0, min_freq_tolerance_hz=1000.0)
    wide = _det("appearance", ts=100.0, center_hz=915e6, bw_hz=200e3)
    drift = _det("appearance", ts=110.0, center_hz=915.080e6, bw_hz=10e3)
    assert tr.observe(wide) is not None
    # 80 kHz away, tolerance = max(wider_bw/2, 1k) = 100 kHz → match
    assert tr.observe(drift) is None


def test_expired_events_are_dropped():
    tr = EventTracker(re_observation_s=30.0)
    a = _det("appearance", ts=100.0)
    tr.observe(a)
    # Advance time past the window. observe() of any new detection
    # triggers expiry.
    far_later = _det("appearance", ts=200.0)
    tr.observe(far_later)
    # Only the new event should remain.
    assert len(tr.all_events()) == 1
    assert tr.all_events()[0].id == far_later.id


def test_disappearance_with_no_prior_state_publishes_and_tracks():
    tr = EventTracker(re_observation_s=30.0)
    # Bin promoted at warmup → disappearance arrives without an
    # earlier appearance.
    d = _det("disappearance", ts=100.0)
    out = tr.observe(d)
    assert out is not None
    evts = tr.all_events()
    assert len(evts) == 1
    assert evts[0].state == "silent"


def test_duplicate_disappearance_within_window_suppressed():
    tr = EventTracker(re_observation_s=30.0)
    d1 = _det("disappearance", ts=100.0)
    d2 = _det("disappearance", ts=105.0)
    assert tr.observe(d1) is not None
    assert tr.observe(d2) is None
