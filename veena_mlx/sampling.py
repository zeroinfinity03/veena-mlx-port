"""Choosing the next token the way the official route does.

The model card calls `generate(do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.05)`.
Run on the real code, that builds four steps in this order, on float32 scores (measured 2026-10-12, see
NOTES.md section 3): repetition penalty, temperature, top-k 50 (a library default the card never
mentions), top-p. Each step below follows its class in repo/transformers/logits_process.py.
"""
import numpy as np

NEG_INF = np.float32(-np.inf)


def repetition_penalty(scores, input_ids, penalty):
    """RepetitionPenaltyLogitsProcessor: every id seen so far, the prompt included."""
    scores = scores.copy()
    ids = np.unique(np.asarray(input_ids, dtype=np.int64))
    score = scores[ids]
    scores[ids] = np.where(score < 0, score * np.float32(penalty), score / np.float32(penalty))
    return scores


def filtered_scores(scores, input_ids, penalty=1.05, temperature=0.4, top_k=50, top_p=0.9):
    """The four steps. scores: float32 [vocab], the logits for the next token; input_ids: every id so far.
    Returns float32 [vocab] with -inf for the tokens that cannot be drawn."""
    # `generate` builds a step, and checks its argument, only when that argument is set and is not the
    # value that does nothing (`_get_logits_processor` in transformers' generation/utils.py)
    use_penalty = penalty is not None and penalty != 1.0
    use_temperature = temperature is not None and temperature != 1.0
    use_top_k = top_k is not None and top_k != 0
    use_top_p = top_p is not None and top_p < 1.0
    if use_penalty and not penalty > 0:
        raise ValueError(f"the repetition penalty has to be a strictly positive number, got {penalty}")
    if use_temperature and not temperature > 0:
        raise ValueError(f"temperature has to be a strictly positive number, got {temperature}")
    if use_top_k and top_k < 0:
        raise ValueError(f"top_k has to be a positive whole number, got {top_k}")
    if use_top_p and top_p < 0:
        raise ValueError(f"top_p has to be between 0 and 1, got {top_p}")
    s = np.asarray(scores, dtype=np.float32)
    if use_penalty:
        s = repetition_penalty(s, input_ids, penalty)
    # TemperatureLogitsWarper
    if use_temperature:
        s = s / np.float32(temperature)
    # TopKLogitsWarper: drop everything strictly below the k-th largest (values tied with it stay)
    if use_top_k:
        k = min(top_k, s.size)
        kth = np.partition(s, s.size - k)[s.size - k]
        s = np.where(s < kth, NEG_INF, s)
    if not use_top_p:
        return s
    # TopPLogitsWarper: sort ascending, softmax, running sum; drop where the sum is <= 1 - top_p; always
    # keep the largest. Tokens already at -inf have probability 0 and stay dropped, so only the finite
    # ones need sorting.
    # Equal scores: the sort is stable, so of several tokens with exactly the same score the lower ids are
    # cut first. bf16 logits tie all the time; `torch.sort` is stable on Apple's GPU (the same sets as here)
    # and not on the CPU, so the official code itself has no single answer for which of two equal tokens
    # stays. The scores that stay are the same either way.
    alive = np.flatnonzero(s > NEG_INF)
    order = alive[np.argsort(s[alive], kind="stable")]
    sorted_logits = s[order]
    # as torch does it: exp times the reciprocal of the sum, and a running sum kept in float64 with every
    # element rounded to float32
    e = np.exp(sorted_logits - sorted_logits[-1])
    probs = (e * (np.float32(1) / e.sum(dtype=np.float32))).astype(np.float32)
    remove = np.cumsum(probs, dtype=np.float64).astype(np.float32) <= np.float32(1 - top_p)
    remove[-1] = False
    s = s.copy()
    s[order[remove]] = NEG_INF
    return s


def probabilities(filtered):
    """softmax of the filtered scores, on the tokens that are left: (token ids, probabilities)."""
    ids = np.flatnonzero(filtered > NEG_INF)
    e = np.exp((filtered[ids] - filtered[ids].max()).astype(np.float64))
    return ids, e / e.sum()


def sample(scores, input_ids, rng, **settings):
    """One multinomial draw from the filtered distribution."""
    ids, p = probabilities(filtered_scores(scores, input_ids, **settings))
    return int(rng.choice(ids, p=p))


def greedy(scores, input_ids, penalty=1.05):
    """`generate(do_sample=False, repetition_penalty=...)`: the penalty, then the best token (lowest id on a tie)."""
    s = np.asarray(scores, dtype=np.float32)
    if penalty is not None and penalty != 1.0:
        if not penalty > 0:
            raise ValueError(f"the repetition penalty has to be a strictly positive number, got {penalty}")
        s = repetition_penalty(s, input_ids, penalty)
    return int(np.argmax(s))
