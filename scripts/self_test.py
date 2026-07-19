"""
self_test.py — sanity checks for the event-level metrics.

Under the refractory rule, FPR/h is physically bounded, whereas the window-level
count is not.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from seizure_metrics import (  # noqa: E402
    evaluate_event_level,
    false_alarms_per_hour,
    generate_alarms,
    infer_seizure_groups,
)

STEP_SEC = 10  # 20 s window, 50% overlap
K, M, REFRACTORY = 5, 12, 180  # thesis alarm params (30-min refractory)


def test_fpr_is_bounded_by_refractory():
    """Even with every window positive, FPR/h cannot exceed ~2 with a 30-min
    refractory. A window-level count would report 360/h here."""
    n = 360 * 3  # 3 hours of interictal at 10 s step
    probs = np.ones(n)  # worst case: model predicts preictal on every window
    y_true = np.zeros(n, dtype=int)  # all interictal
    alarms = generate_alarms(probs, 0.5, K, M, REFRACTORY)
    fpr_h = false_alarms_per_hour(alarms, y_true, STEP_SEC)
    old_window_count = ((probs >= 0.5) & (y_true == 0)).sum() / (n * STEP_SEC / 3600)
    print(f"  worst-case event-level FPR/h = {fpr_h:.2f}  (window-level = {old_window_count:.1f})")
    assert fpr_h <= 2.1, fpr_h
    assert old_window_count > 300  # window-level count is unbounded


def test_event_sensitivity():
    """A run of positive windows before a seizure that triggers K-of-M should
    count as one anticipated seizure."""
    y_true = np.zeros(400, dtype=int)
    y_true[200:230] = 1  # one seizure's preictal window (30 windows)
    probs = np.zeros(400)
    probs[205:220] = 0.9  # confident preictal run inside the window
    groups = infer_seizure_groups(y_true)
    m = evaluate_event_level(y_true, probs, groups, STEP_SEC, 0.5, K, M, REFRACTORY)
    print(f"  event sensitivity = {m.event_sensitivity:.2f} over {m.n_seizures} seizure(s)")
    assert m.n_seizures == 1
    assert m.event_sensitivity == 1.0


def test_no_false_alarm_when_quiet():
    y_true = np.zeros(400, dtype=int)
    probs = np.zeros(400)  # model never crosses threshold
    alarms = generate_alarms(probs, 0.5, K, M, REFRACTORY)
    assert alarms.sum() == 0
    print("  quiet model -> 0 alarms  OK")


if __name__ == "__main__":
    print("Running corrected-metric self-tests...")
    test_fpr_is_bounded_by_refractory()
    test_event_sensitivity()
    test_no_false_alarm_when_quiet()
    print("All self-tests passed.")
