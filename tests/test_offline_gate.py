"""The offline rail check fails for the reasons it claims to check."""

from __future__ import annotations

from scripts.offline_gate import evaluate


def report(unguarded_asr=1.0, guarded_asr=0.5, errors=0, golden_errors=0, frr=0.0, pii=None):
    return {
        "unguarded": {"attack_success_rate": unguarded_asr, "errors": errors, "scored": 72},
        "guarded": {"attack_success_rate": guarded_asr, "errors": 0, "scored": 72},
        "golden": {"errors": golden_errors, "false_refusal_rate": frr, "pii_placeholder_ids": pii or []},
    }


EXPECTED = {"guarded_attack_success_rate": 0.5}


def test_clean_report_passes():
    assert evaluate(report(), EXPECTED, 0.10) == []


def test_weaker_rail_fails():
    assert any("a rail got weaker" in f for f in evaluate(report(guarded_asr=0.6), EXPECTED, 0.10))


def test_stub_not_breaching_means_broken_check():
    assert any("check is broken" in f for f in evaluate(report(unguarded_asr=0.9), EXPECTED, 0.10))


def test_errors_fail():
    assert evaluate(report(errors=1), EXPECTED, 0.10)
    assert evaluate(report(golden_errors=1), EXPECTED, 0.10)


def test_pii_garbling_and_frr_fail():
    assert any("garbled" in f for f in evaluate(report(pii=["g061"]), EXPECTED, 0.10))
    assert any("refused" in f for f in evaluate(report(frr=0.2), EXPECTED, 0.10))
