"""
Production dataset builder for the sliding-block puzzle generator.

Goal: given a TOTAL number of games you want (e.g. 1,000,000), automatically:
  1. split it into 5 difficulty tiers (even split by default, e.g. 200k each)
  2. OVERSHOOT each tier by some factor (default 1.5x) using fast random
     generation + the solver-agnostic difficulty_score to bucket candidates
  3. validate solvability (cheap bounded BFS) only on candidates that already
     landed in the right difficulty bucket -- avoids wasting solver time on
     boards you'll throw away anyway
  4. randomly subsample each tier's oversupply down to the exact target count
  5. stream everything to disk as JSONL (one board per line) so memory never
     holds the full dataset, and write a manifest summarizing the run

Run small first (--total 5000) to sanity check, then scale up (--total 1000000)
across multiple processes / machines as needed -- it's embarrassingly parallel,
each worker just needs a different random seed.
"""

import random
import json
import argparse
import multiprocessing as mp
import time
import os

from puzzle_core import (
    random_board, random_board_by_occupancy, board_occupancy,
    compute_difficulty_score, solve_bfs, board_to_dict, board_signature,
)

# ---------------------------------------------------------------------------
# Tier boundaries -- calibrated empirically from the difficulty_score
# distribution of randomly generated boards (see calibration step below).
# These are quantile cut points so each tier gets roughly equal representation
# from naive random generation, before any tier-targeted oversampling.
# ---------------------------------------------------------------------------

DEFAULT_TIER_EDGES = [25.0, 38.0, 43.5, 49.5]  # 4 edges -> 5 tiers
TIER_NAMES = ["very_easy", "easy", "medium", "hard", "very_hard"]

# ---------------------------------------------------------------------------
# Per-tier occupancy tilt: easier tiers skew toward the LOW end of the
# [12, 26] occupancy band (more open board) and get a favorable, clearer
# starting lane in front of the red block; harder tiers skew toward the HIGH
# end (dense/cluttered) with no favorability bias (pure random placement).
# Ranges deliberately overlap so the solver-based difficulty_score is still
# what ultimately decides which tier a board lands in -- occupancy just
# biases the *distribution* of candidates each tier draws from.
# ---------------------------------------------------------------------------

TIER_OCCUPANCY_RANGES = {
    "very_easy": (12, 17),
    "easy":      (15, 20),
    "medium":    (18, 23),
    "hard":      (21, 26),
    "very_hard": (23, 26),
}

TIER_RED_ROW_AVOID_PROB = {
    "very_easy": 0.85,
    "easy":      0.55,
    "medium":    0.25,
    "hard":      0.0,
    "very_hard": 0.0,
}


def score_to_tier(score, edges=DEFAULT_TIER_EDGES):
    for i, e in enumerate(edges):
        if score < e:
            return i
    return len(edges)


def calibrate_tier_edges(sample_size=800, num_blocks_range=(5, 9), n_tiers=5, seed=None):
    """Optional: re-derive tier boundaries from a fresh random sample instead
    of using the hardcoded defaults. Run this once if your board parameters
    (grid size, block count range, etc.) change."""
    rng = random.Random(seed)
    scores = []
    while len(scores) < sample_size:
        nb = rng.randint(*num_blocks_range)
        b = random_board(num_blocks=nb)
        if b is None:
            continue
        scores.append(compute_difficulty_score(b)["difficulty_score"])
    scores.sort()
    step = len(scores) // n_tiers
    edges = [scores[min(i * step, len(scores) - 1)] for i in range(1, n_tiers)]
    return edges


# ---------------------------------------------------------------------------
# Worker: generate boards for ONE tier until it collects `quota` valid ones.
# Each worker process gets its own RNG seed (passed in) so parallel workers
# don't duplicate effort.
# ---------------------------------------------------------------------------

def _tier_worker(args):
    (tier_idx, quota, seed, num_blocks_range, tier_edges,
     max_solver_states, out_path, min_occupancy, max_occupancy) = args
    random.seed(seed)
    lo = tier_edges[tier_idx - 1] if tier_idx > 0 else -1
    hi = tier_edges[tier_idx] if tier_idx < len(tier_edges) else float("inf")
    tier_name = TIER_NAMES[tier_idx]

    # if the caller didn't override min/max occupancy, fall back to this
    # tier's own skewed range (easy tiers -> low occupancy, hard -> high)
    if min_occupancy is None or max_occupancy is None:
        t_min, t_max = TIER_OCCUPANCY_RANGES[tier_name]
        min_occupancy = t_min if min_occupancy is None else min_occupancy
        max_occupancy = t_max if max_occupancy is None else max_occupancy
    avoid_prob = TIER_RED_ROW_AVOID_PROB[tier_name]

    collected = 0
    tries = 0
    with open(out_path, "w") as f:
        while collected < quota:
            tries += 1
            # occupancy-driven placement, tilted per-tier: keeps dropping
            # length-2/3 blocks until total covered cells lands in this
            # tier's [min_occupancy, max_occupancy] slice of the [12, 26]
            # band, with red_row_avoid_prob keeping the escape lane clearer
            # for easier tiers (favorable start) and unbiased for hard ones.
            blocks = random_board_by_occupancy(
                min_occupancy=min_occupancy, max_occupancy=max_occupancy,
                red_row_avoid_prob=avoid_prob)
            if blocks is None or len(blocks) < 3:
                continue

            # cheap geometric check FIRST (no solver call) -- only boards
            # that already land in this tier's score range proceed to the
            # expensive solvability check
            diff = compute_difficulty_score(blocks)
            if not (lo <= diff["difficulty_score"] < hi):
                continue

            moves = solve_bfs(blocks, max_states=max_solver_states)
            if moves is None:
                continue  # unsolvable within budget, discard

            record = {
                "tier": tier_name,
                "tier_idx": tier_idx,
                "min_solve_moves": moves,
                "occupancy": board_occupancy(blocks),
                "board": board_to_dict(blocks),
            }
            f.write(json.dumps(record) + "\n")
            collected += 1

    return tier_idx, collected, tries, out_path


def generate_tier_pool(tier_idx, quota, workers, num_blocks_range, tier_edges,
                        max_solver_states, work_dir, min_occupancy=None, max_occupancy=None):
    """Spread one tier's quota across multiple worker processes, each writing
    its own shard file, then return the list of shard paths.

    min_occupancy/max_occupancy: pass None (default) to let each tier use its
    own skewed range from TIER_OCCUPANCY_RANGES; pass explicit ints to force
    the same fixed occupancy band across every tier instead."""
    per_worker = quota // workers + 1
    jobs = []
    for w in range(workers):
        out_path = os.path.join(work_dir, f"tier{tier_idx}_shard{w}.jsonl")
        jobs.append((tier_idx, per_worker, random.randint(0, 2**31),
                     num_blocks_range, tier_edges, max_solver_states, out_path,
                     min_occupancy, max_occupancy))

    shard_paths = []
    total_collected = 0
    total_tries = 0
    with mp.Pool(workers) as pool:
        for t_idx, collected, tries, path in pool.imap_unordered(_tier_worker, jobs):
            shard_paths.append(path)
            total_collected += collected
            total_tries += tries
    return shard_paths, total_collected, total_tries


# ---------------------------------------------------------------------------
# Top-level driver: split total -> 5 tiers, oversample, subsample, merge.
# ---------------------------------------------------------------------------

def build_dataset(total_games, output_path, n_tiers=5, oversample_factor=1.5,
                   workers=8, num_blocks_range=(5, 9), tier_edges=None,
                   max_solver_states=150000, work_dir="_dataset_shards",
                   seed=None, min_occupancy=None, max_occupancy=None):
    if seed is not None:
        random.seed(seed)
    os.makedirs(work_dir, exist_ok=True)
    # ensure the OUTPUT file's parent directory exists too -- open() does
    # not create missing parent folders, so pointing --out at a not-yet-
    # existing folder would otherwise raise FileNotFoundError
    out_parent = os.path.dirname(os.path.abspath(output_path))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    tier_edges = tier_edges or DEFAULT_TIER_EDGES

    per_tier_target = total_games // n_tiers
    remainder = total_games - per_tier_target * n_tiers  # give leftover to last tier
    per_tier_targets = [per_tier_target] * n_tiers
    per_tier_targets[-1] += remainder

    manifest = {"total_requested": total_games, "tiers": []}
    final_count = 0
    t_start = time.time()

    with open(output_path, "w") as out_f:
        for tier_idx in range(n_tiers):
            target = per_tier_targets[tier_idx]
            oversample_quota = int(target * oversample_factor)
            t0 = time.time()

            shard_paths, collected, tries = generate_tier_pool(
                tier_idx, oversample_quota, workers, num_blocks_range,
                tier_edges, max_solver_states, work_dir,
                min_occupancy=min_occupancy, max_occupancy=max_occupancy)

            # load this tier's oversupply, dedupe by board geometry, then
            # randomly subsample to exact target (dedup BEFORE sampling so
            # duplicates don't silently eat into your target count)
            pool = []
            for sp in shard_paths:
                with open(sp) as sf:
                    pool.extend(sf.readlines())
                os.remove(sp)  # cleanup shard

            seen_sigs = set()
            deduped_pool = []
            dupes_dropped = 0
            for line in pool:
                rec = json.loads(line)
                sig = board_signature(rec["board"])
                if sig in seen_sigs:
                    dupes_dropped += 1
                    continue
                seen_sigs.add(sig)
                deduped_pool.append(line)
            pool = deduped_pool

            take = min(target, len(pool))
            sampled = random.sample(pool, take)
            for line in sampled:
                out_f.write(line)
            final_count += take

            elapsed = time.time() - t0
            manifest["tiers"].append({
                "tier": TIER_NAMES[tier_idx],
                "target": target,
                "oversampled_pool_size": collected,
                "duplicates_dropped": dupes_dropped,
                "total_attempts": tries,
                "final_count": take,
                "acceptance_rate": round(collected / tries, 4) if tries else 0,
                "seconds": round(elapsed, 1),
            })
            print(f"[{TIER_NAMES[tier_idx]:10s}] target={target:>7d}  "
                  f"oversampled={collected:>7d}  dupes={dupes_dropped:>4d}  "
                  f"final={take:>7d}  ({elapsed:.1f}s, {collected/tries:.1%} accept)")

    manifest["final_total"] = final_count
    manifest["total_seconds"] = round(time.time() - t_start, 1)
    manifest_path = output_path.replace(".jsonl", "_manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    os.rmdir(work_dir) if os.path.isdir(work_dir) and not os.listdir(work_dir) else None
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2000,
                         help="total games wanted across all 5 tiers")
    parser.add_argument("--oversample", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=str, default="./dataset_sample.jsonl")
    parser.add_argument("--calibrate", action="store_true",
                         help="re-derive tier edges from a fresh sample instead of defaults")
    parser.add_argument("--min-occupancy", type=int, default=None,
                         help="force a fixed min occupancy across all tiers "
                              "(default: None, each tier uses its own skewed range)")
    parser.add_argument("--max-occupancy", type=int, default=None,
                         help="force a fixed max occupancy across all tiers "
                              "(default: None, each tier uses its own skewed range)")
    args = parser.parse_args()

    edges = DEFAULT_TIER_EDGES
    if args.calibrate:
        print("Calibrating tier edges from a fresh sample...")
        edges = calibrate_tier_edges()
        print("New tier edges:", edges)

    manifest = build_dataset(
        total_games=args.total,
        output_path=args.out,
        oversample_factor=args.oversample,
        workers=args.workers,
        tier_edges=edges,
        min_occupancy=args.min_occupancy,
        max_occupancy=args.max_occupancy,
    )
    print("\nDone. Manifest:")
    print(json.dumps(manifest, indent=2))
