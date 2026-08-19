"""
Community Alignment -> LoRe-format preference pairs, filtered for English and data-rich users.

The goal: "Find another multi-user preference dataset where there's more data per user (Community
Alignment, filtered for english language and users with > some large number of datapoints each).
See if you can modify LoRe to get it to work on there."

Why the threshold matters: synthetic_recovery.py shows LoRe has a sharp phase transition around
~50 pairs/user -- below ~25 it cannot recover even clean, orthogonal, planted axes, and above ~100 it
recovers them near-perfectly. PRISM had ~15 pairs/user, which is why it nulled out. So the filter here
is not arbitrary: --min_pairs should be >= 50, ideally >= 100.

Dataset shape (facebook/community-alignment-dataset, 90,256 rows):
  * annotator_id     -> the user
  * assigned_lang    -> language code; English is 'en' (NOT 'English'), 30,856 rows
  * {first,second,third,fourth}_turn_response_{a,b,c,d} + _preferred_response
    Each turn shows 4 responses and records which was preferred => 3 pairs per turn
    (preferred vs each of the other 3), up to 4 turns per conversation.

Output: a JSON list of {user_id, prompt, chosen, rejected, turn, conversation_id}, ready to embed
with the same last-token-hidden-state pipeline used for PRISM. Deliberately mirrors PRISM's structure
so the existing diagnostics (per_user_signal.py, cluster_signal.py, multiseed_collapse.py) apply
unchanged.

IMPORTANT (lesson from the PRISM string-vs-list bug): chosen and rejected are BOTH plain strings
here, formatted identically, and we assert that before writing. That artifact is exactly what made
LoRe's PRISM numbers meaningless, so we check rather than assume.

Usage:
  python community_alignment_prep.py --stats                    # just report the per-user distribution
  python community_alignment_prep.py --min_pairs 100 --out data/community_alignment/pairs.json
"""
import argparse
import json
import os
import sys

TURNS = ["first", "second", "third", "fourth"]
LETTERS = ["a", "b", "c", "d"]


def _clean(x):
    """Normalize a cell to a stripped string, or None if effectively empty."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", ""}:
        return None
    return s


def load_dataframe(cache_dir, columns=None):
    """Load from the cached parquet if present, else pull from the Hub."""
    import pandas as pd
    cached = os.path.join(cache_dir, "meta_cols.parquet")
    if columns is not None and os.path.exists(cached):
        return pd.read_parquet(cached)
    from datasets import load_dataset
    ds = load_dataset("facebook/community-alignment-dataset", split="train")
    return ds.to_pandas()


def count_pairs_per_turn(row):
    """How many valid pairs this row's turns yield (3 per turn that has a preferred response)."""
    n = 0
    for t in TURNS:
        if _clean(row.get(f"{t}_turn_preferred_response")) is not None:
            n += 3
    return n


def user_pair_counts(df):
    """Series: annotator_id -> total pairs available (English rows only)."""
    en = df[df["assigned_lang"] == "en"].copy()
    en["pairs"] = en.apply(count_pairs_per_turn, axis=1)
    return en.groupby("annotator_id")["pairs"].sum().sort_values(ascending=False), en


def report_stats(counts, en):
    import numpy as np
    n_turns = en.apply(lambda r: count_pairs_per_turn(r) // 3, axis=1)
    print(f"English conversations : {len(en)}")
    print(f"English annotators    : {len(counts)}")
    print("\nturns per conversation:")
    print(n_turns.value_counts().sort_index().to_string())
    print(f"\npairs/user  median {int(counts.median())}  mean {counts.mean():.1f}  max {int(counts.max())}")
    print(f"\n{'threshold':>9} | {'users':>6} | {'pairs from them':>15}")
    print("-" * 38)
    for thr in (15, 25, 50, 100, 200, 400):
        sel = counts[counts >= thr]
        print(f"{thr:>9} | {len(sel):>6} | {int(sel.sum()):>15}")
    print("\n(synthetic_recovery.py: LoRe needs >=50 pairs/user, ideally >=100. "
          "PRISM had ~15 and nulled out.)")


def build_pairs(en, counts, min_pairs, max_users=None, max_pairs_per_user=None, seed=0):
    """Emit (user, prompt, chosen, rejected) for users clearing the threshold.

    max_users / max_pairs_per_user bound the embedding cost: every pair needs its two sides embedded
    through an 8B model, so the full set (941 users x ~200 pairs) would be ~250k forward passes.
    Users are sampled at random (not top-N by count) so the sample is not biased toward the most
    prolific annotators.
    """
    eligible = counts[counts >= min_pairs]
    if max_users and len(eligible) > max_users:
        eligible = eligible.sample(n=max_users, random_state=seed)
    keep = set(eligible.index)
    per_user = {}
    out = []
    for _, row in en.iterrows():
        uid = row["annotator_id"]
        if uid not in keep:
            continue
        for t in TURNS:
            pref = _clean(row.get(f"{t}_turn_preferred_response"))
            prompt = _clean(row.get(f"{t}_turn_prompt"))
            if pref is None or prompt is None:
                continue
            responses = {L: _clean(row.get(f"{t}_turn_response_{L}")) for L in LETTERS}
            # preferred_response is stored as "response_a".."response_d" (verified against the
            # data: value_counts gives response_d 31179 / response_a 26849 / ...), so map it back
            # to the matching response column. Anything else is unusable, so skip the turn.
            letter = pref.lower().replace("response_", "").strip()
            chosen = responses.get(letter)
            if chosen is None:
                continue
            for L, text in responses.items():
                if text is None or text == chosen:
                    continue
                if max_pairs_per_user and per_user.get(uid, 0) >= max_pairs_per_user:
                    break
                per_user[uid] = per_user.get(uid, 0) + 1
                # Guard against the PRISM formatting artifact: both sides must be plain strings.
                assert isinstance(chosen, str) and isinstance(text, str), \
                    "chosen/rejected must both be plain strings (see the PRISM list-vs-string bug)"
                out.append({
                    "user_id": str(uid),
                    "conversation_id": str(row.get("conversation_id")),
                    "turn": t,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": text,
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "community_alignment"))
    ap.add_argument("--min_pairs", type=int, default=100,
                    help="keep users with at least this many pairs (>=50 per synthetic_recovery.py)")
    ap.add_argument("--max_users", type=int, default=None,
                    help="randomly sample at most this many eligible users (bounds embedding cost)")
    ap.add_argument("--max_pairs_per_user", type=int, default=None,
                    help="cap pairs kept per user (bounds embedding cost)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="report the distribution and exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta_cols = ["annotator_id", "assigned_lang", "conversation_id"] + \
                [f"{t}_turn_preferred_response" for t in TURNS]
    df = load_dataframe(args.cache_dir, columns=meta_cols if args.stats else None)
    counts, en = user_pair_counts(df)

    report_stats(counts, en)
    if args.stats:
        return

    if "first_turn_prompt" not in en.columns:
        print("\n[!] Cached metadata lacks the prompt/response text columns, so pairs cannot be "
              "built from it. Re-run without --stats once the full dataset is available locally.",
              file=sys.stderr)
        sys.exit(2)

    pairs = build_pairs(en, counts, args.min_pairs, args.max_users,
                        args.max_pairs_per_user, args.seed)
    out = args.out or os.path.join(args.cache_dir, f"pairs_min{args.min_pairs}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(pairs, f)
    users = len({p["user_id"] for p in pairs})
    print(f"\nWrote {len(pairs)} pairs from {users} users (min_pairs={args.min_pairs}) -> {out}")


if __name__ == "__main__":
    main()
