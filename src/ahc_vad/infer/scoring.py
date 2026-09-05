"""Turn a VLM generation into a (class_name, score) verdict with a REAL continuous score.

Why this module exists: the Stage-3 aggregator thresholds window scores with hysteresis
(theta_high to open an event, theta_low to sustain it). Parsing the decoded JSON string
alone yields no continuous quantity - every window would score 1.0 and both thresholds
would be inert. See docs/architecture.md Stage 2.

Where the score comes from: the model emits {"is_anomaly": true|false, "class_name": ...}.
`true` and `false` are single tokens, so at that one decode position the softmax gives a
direct, calibrated P(anomaly) for the window. Class identity is taken from greedy decoding.

Why not a full 12-way class distribution: the class names share token prefixes
(`traffic_accident` vs `traffic_congestion` both start `traffic`), so the first generated
class token cannot separate them. Getting true per-class probabilities would need either
12 teacher-forced continuations (one extra batched forward per window - real cost against
a real-time budget) or a single-token label vocabulary (which would mean regenerating the
training data). P(anomaly) is the quantity the aggregator actually gates on, so this buys
working hysteresis for one forward pass.
"""

from __future__ import annotations

import json

import torch

from ahc_vad.schema import CLASS_NAMES

TRUE_STRINGS = ("true", "True")
FALSE_STRINGS = ("false", "False")


def anomaly_probability(processor, generated_ids, scores) -> float | None:
    """P(anomaly) read at the boolean decode position.

    `scores` is the per-step logits tuple from generate(..., output_scores=True).
    Returns None when the boolean token cannot be located, so callers can fall back
    rather than silently trusting a fabricated number.
    """
    if not scores:
        return None

    for step, token_id in enumerate(generated_ids):
        if step >= len(scores):
            break
        piece = processor.decode([token_id], skip_special_tokens=True).strip()
        if not piece:
            continue

        is_true = any(piece.startswith(s) for s in TRUE_STRINGS)
        is_false = any(piece.startswith(s) for s in FALSE_STRINGS)
        if not (is_true or is_false):
            continue

        probs = torch.softmax(scores[step][0].float(), dim=-1)

        # Compare the two competing tokens directly and renormalise over just that pair.
        # Absolute softmax mass is spread over the whole vocabulary; what we want is the
        # model's confidence in true-vs-false specifically.
        true_ids = set(_token_ids_for(processor, TRUE_STRINGS))
        false_ids = set(_token_ids_for(processor, FALSE_STRINGS))

        # A token id landing in BOTH sets would be counted on both sides and silently skew
        # the ratio (equal logits would not give 0.5). Drop any such id rather than trust it.
        overlap = true_ids & false_ids
        if overlap:
            true_ids -= overlap
            false_ids -= overlap
        if not true_ids or not false_ids:
            return 1.0 if is_true else 0.0  # cannot separate; fall back to the hard decision

        vocab = probs.shape[0]
        p_true = float(sum(probs[i].item() for i in true_ids if i < vocab))
        p_false = float(sum(probs[i].item() for i in false_ids if i < vocab))

        total = p_true + p_false
        if total <= 0:
            return 1.0 if is_true else 0.0
        return p_true / total

    return None


_TOKEN_ID_CACHE: dict[int, dict[str, list[int]]] = {}


def _token_ids_for(processor, strings: tuple[str, ...]) -> list[int]:
    """First-token ids for candidate strings, with and without a leading space."""
    tok = getattr(processor, "tokenizer", processor)
    key = id(tok)
    cache = _TOKEN_ID_CACHE.setdefault(key, {})
    cache_key = "|".join(strings)
    if cache_key in cache:
        return cache[cache_key]

    ids: set[int] = set()
    for s in strings:
        for variant in (s, f" {s}"):
            try:
                encoded = tok.encode(variant, add_special_tokens=False)
            except TypeError:
                encoded = tok.encode(variant)
            if encoded:
                ids.add(encoded[0])
    out = sorted(ids)
    cache[cache_key] = out
    return out


def parse_class(text: str) -> tuple[str, bool]:
    """Extract (class_name, is_anomaly) from the model's JSON reply.

    Falls back to 'normal' on malformed output rather than raising - a single unparseable
    window should not abort a whole video.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return "normal", False

    class_name = obj.get("class_name")
    is_anomaly = bool(obj.get("is_anomaly", False))
    if class_name not in CLASS_NAMES:
        return "normal", False
    if not is_anomaly or class_name == "normal":
        return "normal", False
    return class_name, True


def verdict_from_generation(processor, generated_ids, scores, text: str) -> tuple[str, float]:
    """Combine greedy class identity with the calibrated anomaly probability."""
    class_name, is_anomaly = parse_class(text)
    p = anomaly_probability(processor, generated_ids, scores)

    if p is None:
        # No usable score. Fall back to the hard decision rather than inventing a value -
        # this keeps behaviour correct, though hysteresis degrades to a step function.
        return class_name, (1.0 if is_anomaly else 0.0)

    # `p` is P(is_anomaly=true). A window the model called normal scores 0 for every
    # anomaly class, which is exactly what the aggregator needs to close open spans.
    return (class_name, p) if is_anomaly else ("normal", 0.0)
