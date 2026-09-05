"""Tests for the window-scoring logic (src/ahc_vad/infer/scoring.py).

The score feeds the Stage-3 aggregator's hysteresis thresholds, so getting the
true-vs-false probability right matters: if it is miscalibrated, theta_high/theta_low
gate on a meaningless number.

Run: python tests/test_scoring.py
"""

import torch

from ahc_vad.infer.scoring import anomaly_probability, parse_class, verdict_from_generation

# A realistic tokenizer distinguishes 'true'/'True' as separate tokens.
VOCAB = ["{", '"is_anomaly"', ":", "true", "false", ",", '"class_name"', '"fire"', "}",
         " ", "True", "False"]
ID = {t: i for i, t in enumerate(VOCAB)}


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        s2 = s.strip()
        if s2 not in ID:
            return []
        return [ID[s2]]


class FakeProc:
    tokenizer = FakeTok()

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(VOCAB[i] for i in ids)


proc = FakeProc()
BOOL_POS = 3


def make_seq(bool_token):
    return [ID["{"], ID['"is_anomaly"'], ID[":"], ID[bool_token], ID[","],
            ID['"class_name"'], ID['"fire"'], ID["}"]]


def make_scores(seq, true_logit, false_logit):
    """Per-step logits; the boolean position carries the true/false contrast."""
    out = []
    for i, _ in enumerate(seq):
        logits = torch.full((1, len(VOCAB)), -10.0)
        if i == BOOL_POS:
            logits[0, ID["true"]] = true_logit
            logits[0, ID["false"]] = false_logit
        out.append(logits)
    return tuple(out)


def main():
    failures = []

    def check(name, got, expect, tol=None):
        ok = abs(got - expect) < tol if tol is not None else got == expect
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: got={got if tol is None else f'{got:.3f}'}"
              f" expect={expect}")
        if not ok:
            failures.append(name)

    # The decoded token must agree with the argmax, as greedy decoding guarantees.
    print("=== anomaly_probability: calibration ===")
    for name, tl, fl, tok, expect in [
        ("confident true",  5.0, -5.0, "true",  0.9999),
        ("confident false", -5.0, 5.0, "false", 0.0001),
        ("uncertain",        0.0, 0.0, "true",  0.5),
        ("mild true",        1.0, 0.0, "true",  0.7311),
        ("mild false",       0.0, 1.0, "false", 0.2689),
    ]:
        seq = make_seq(tok)
        p = anomaly_probability(proc, seq, make_scores(seq, tl, fl))
        check(name, p, expect, tol=0.01)

    print("\n=== must not fabricate a score when none is available ===")
    seq = make_seq("true")
    p = anomaly_probability(proc, seq, ())
    print(f"  {'OK  ' if p is None else 'FAIL'} no scores -> {p}")
    if p is not None:
        failures.append("no-scores returns None")

    print("\n=== parse_class ===")
    for text, expect in [
        ('{"is_anomaly":true,"class_name":"fire"}', ("fire", True)),
        ('{"is_anomaly":false,"class_name":"normal"}', ("normal", False)),
        ("garbage not json", ("normal", False)),
        ('{"is_anomaly":true,"class_name":"not_a_class"}', ("normal", False)),
        ('noise {"is_anomaly":true,"class_name":"smoke"} trailing', ("smoke", True)),
    ]:
        check(text[:44], parse_class(text), expect)

    print("\n=== verdict_from_generation ===")
    # A normal window must score exactly 0 for every class: that is what closes an open
    # span in the aggregator. A non-zero normal score would merge distinct events.
    seq = make_seq("false")
    got = verdict_from_generation(proc, seq, make_scores(seq, -5.0, 5.0),
                                  '{"is_anomaly":false,"class_name":"normal"}')
    check("normal window scores 0", got, ("normal", 0.0))

    seq = make_seq("true")
    cls, sc = verdict_from_generation(proc, seq, make_scores(seq, 3.0, -3.0),
                                      '{"is_anomaly":true,"class_name":"fire"}')
    ok = cls == "fire" and 0.9 < sc < 1.0
    print(f"  {'OK  ' if ok else 'FAIL'} anomaly window -> ({cls}, {sc:.3f})")
    if not ok:
        failures.append("anomaly verdict")

    # Ordering is what hysteresis relies on: a more confident window must score higher.
    print("\n=== monotonicity (hysteresis depends on this) ===")
    seq = make_seq("true")
    ps = [anomaly_probability(proc, seq, make_scores(seq, tl, 0.0)) for tl in [0.0, 1.0, 3.0, 6.0]]
    ok = all(a < b for a, b in zip(ps, ps[1:]))
    print(f"  {'OK  ' if ok else 'FAIL'} increasing confidence -> {[f'{p:.3f}' for p in ps]}")
    if not ok:
        failures.append("monotonicity")

    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
