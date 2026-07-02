"""
Build the full multi-method dataset:
  - 200k boards x 4 generation methods (random / bfs / astar / idastar) = 800k total
  - PLUS a separate 50k "holdout" set, generated via the RANDOM method, that is
    GUARANTEED to share no board (by exact geometry) with any of the 800k.

Why a guaranteed-disjoint holdout matters: if you're training agents on the
800k and then evaluating generalization, any overlap between train and eval
sets silently inflates your reported performance (the agent could just be
recalling a board it trained on rather than actually solving a new one).
This script tracks every board's canonical signature as it's generated and
rejects holdout candidates that collide with ANYTHING in the 800k pool.

Usage:
    python3 build_full_dataset.py --per-method 200000 --holdout 50000

Defaults are small for a quick sanity-check run; scale up for the real thing.
"""

import random
import json
import argparse
import multiprocessing as mp
import time
import os

from puzzle_core import (
    random_board, random_board_by_occupancy, board_occupancy,
    board_to_dict, board_signature, solve_bfs,
    GENERATORS,
)

METHODS = ["random", "bfs", "astar", "idastar"]


# ---------------------------------------------------------------------------
# Step 1: generate 200k (or N) boards for one method, write to its own file.
# Each record includes its signature so later steps can dedupe/exclude
# without recomputing geometry from scratch.
# ---------------------------------------------------------------------------

def _gen_random_worker(args):
    (seed, quota, num_blocks_range, out_path, min_moves, max_moves,
     min_occupancy, max_occupancy) = args
    random.seed(seed)
    collected = 0
    tries = 0
    with open(out_path, "w") as f:
        while collected < quota:
            tries += 1
            # occupancy-driven placement (mix of length-2/3 pieces, total
            # covered cells within [min_occupancy, max_occupancy]) instead
            # of the old fixed block-count approach.
            blocks = random_board_by_occupancy(
                min_occupancy=min_occupancy, max_occupancy=max_occupancy)
            if blocks is None or len(blocks) < 3:
                continue
            moves = solve_bfs(blocks, max_states=150000)
            if moves is None or moves < min_moves or moves > max_moves:
                continue
            board = board_to_dict(blocks)
            rec = {"method": "random", "min_solve_moves": moves,
                   "occupancy": board_occupancy(blocks), "board": board}
            f.write(json.dumps(rec) + "\n")
            collected += 1
    return out_path, collected, tries


def _gen_scramble_worker(args):
    method, seed, quota, num_blocks, depth_range, out_path = args
    random.seed(seed)
    gen_fn = GENERATORS[method]
    collected = 0
    tries = 0
    with open(out_path, "w") as f:
        while collected < quota:
            tries += 1
            target_depth = random.randint(*depth_range)
            board, depth, path = gen_fn(target_depth=target_depth, num_blocks=num_blocks)
            rec = {
                "method": method,
                "target_depth": target_depth,
                "actual_depth": depth,
                "move_path": path,
                "board": board_to_dict(board),
            }
            f.write(json.dumps(rec) + "\n")
            collected += 1
    return out_path, collected, tries


def generate_method_pool(method, total, workers, work_dir, num_blocks_range=(5, 9),
                          depth_range=(4, 16), min_moves=4, max_moves=30,
                          min_occupancy=12, max_occupancy=26):
    """Generate `total` boards for one method, parallelized, written as
    one JSONL shard per worker. Returns list of shard paths."""
    per_worker = total // workers + 1
    jobs = []
    for w in range(workers):
        out_path = os.path.join(work_dir, f"{method}_shard{w}.jsonl")
        seed = random.randint(0, 2**31)
        if method == "random":
            jobs.append((seed, per_worker, num_blocks_range, out_path, min_moves, max_moves,
                         min_occupancy, max_occupancy))
        else:
            jobs.append((method, seed, per_worker, sum(num_blocks_range) // 2, depth_range, out_path))

    worker_fn = _gen_random_worker if method == "random" else _gen_scramble_worker
    shard_paths = []
    total_collected = 0
    total_tries = 0
    with mp.Pool(workers) as pool:
        for path, collected, tries in pool.imap_unordered(worker_fn, jobs):
            shard_paths.append(path)
            total_collected += collected
            total_tries += tries
    return shard_paths, total_collected, total_tries


# ---------------------------------------------------------------------------
# Step 2: merge all 4 methods' shards into the final 800k file, dedupe
# WITHIN and ACROSS methods, and build the global signature set used to
# guarantee the holdout set is disjoint.
# ---------------------------------------------------------------------------

def merge_and_dedupe(all_shard_paths, final_path, target_per_method, methods=METHODS):
    """Merge shards, dedupe globally (across ALL methods, not just within
    one), trim each method down to its target count, write final file,
    and return the set of all signatures used (for holdout exclusion)."""
    global_seen = set()
    per_method_lines = {m: [] for m in methods}

    for method, paths in all_shard_paths.items():
        for sp in paths:
            with open(sp) as f:
                for line in f:
                    rec = json.loads(line)
                    sig = board_signature(rec["board"])
                    if sig in global_seen:
                        continue  # duplicate, possibly even across methods
                    global_seen.add(sig)
                    per_method_lines[method].append(line)
            os.remove(sp)

    final_count = 0
    with open(final_path, "w") as out_f:
        for m in methods:
            lines = per_method_lines[m]
            take = min(target_per_method, len(lines))
            sampled = random.sample(lines, take)
            for line in sampled:
                out_f.write(line)
            final_count += take
            print(f"  [{m:8s}] pool={len(lines):>7d}  kept={take:>7d}")

    return global_seen, final_count


# ---------------------------------------------------------------------------
# Step 3: generate the holdout set, rejecting any board whose signature is
# already in `global_seen` (the 800k pool). This is the actual disjointness
# guarantee -- not probabilistic, checked exactly per candidate.
# ---------------------------------------------------------------------------

def generate_disjoint_holdout(count, excluded_signatures, out_path,
                               num_blocks_range=(5, 9), min_moves=4, max_moves=30,
                               max_tries=None, min_occupancy=12, max_occupancy=26):
    """Random-method generation only, rejecting anything whose signature
    collides with `excluded_signatures`. Single-process by design, since the
    exclusion set must be checked against EVERY candidate consistently --
    splitting across workers would need shared memory for the exclusion set,
    which is a fine optimization later but not needed at 50k scale."""
    collected = 0
    tries = 0
    collisions_rejected = 0
    max_tries = max_tries or count * 200

    local_seen = set()  # also dedupe WITHIN the holdout set itself

    with open(out_path, "w") as f:
        while collected < count and tries < max_tries:
            tries += 1
            blocks = random_board_by_occupancy(
                min_occupancy=min_occupancy, max_occupancy=max_occupancy)
            if blocks is None or len(blocks) < 3:
                continue
            board = board_to_dict(blocks)
            board["occupancy"] = board_occupancy(blocks)
            sig = board_signature(board)

            if sig in excluded_signatures:
                collisions_rejected += 1
                continue
            if sig in local_seen:
                continue

            moves = solve_bfs(blocks, max_states=150000)
            if moves is None or moves < min_moves or moves > max_moves:
                continue

            local_seen.add(sig)
            rec = {"method": "random_holdout", "min_solve_moves": moves, "board": board}
            f.write(json.dumps(rec) + "\n")
            collected += 1

    return collected, tries, collisions_rejected


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def build_full_dataset(per_method, holdout_count, work_dir, final_path, holdout_path,
                        workers=8, num_blocks_range=(5, 9), depth_range=(4, 16),
                        min_occupancy=12, max_occupancy=26):
    os.makedirs(work_dir, exist_ok=True)
    # also ensure the OUTPUT file directories exist -- open() will not
    # create missing parent folders on its own (this is what caused the
    # FileNotFoundError when --out pointed at a folder that didn't exist yet)
    for p in (final_path, holdout_path):
        parent = os.path.dirname(os.path.abspath(p))
        if parent:
            os.makedirs(parent, exist_ok=True)
    t0 = time.time()

    print(f"=== Step 1: generating ~{per_method} candidates per method (4 methods) ===")
    all_shard_paths = {}
    for method in METHODS:
        # oversample slightly since within/across-method dedup will trim some
        oversample_target = int(per_method * 1.1)
        shard_paths, collected, tries = generate_method_pool(
            method, oversample_target, workers, work_dir,
            num_blocks_range=num_blocks_range, depth_range=depth_range,
            min_occupancy=min_occupancy, max_occupancy=max_occupancy)
        all_shard_paths[method] = shard_paths
        print(f"  {method:8s}: generated {collected} candidates from {tries} attempts "
              f"({time.time()-t0:.1f}s elapsed)")

    print(f"\n=== Step 2: merging + deduping into final {per_method * 4} board dataset ===")
    global_seen, final_count = merge_and_dedupe(all_shard_paths, final_path, per_method)
    print(f"  Final dataset: {final_count} boards -> {final_path}")
    print(f"  Global signature pool size: {len(global_seen)}")

    print(f"\n=== Step 3: generating {holdout_count} DISJOINT holdout boards (random method) ===")
    t1 = time.time()
    h_collected, h_tries, h_rejected = generate_disjoint_holdout(
        holdout_count, global_seen, holdout_path, num_blocks_range=num_blocks_range,
        min_occupancy=min_occupancy, max_occupancy=max_occupancy)
    print(f"  Holdout: {h_collected} boards collected, {h_tries} attempts, "
          f"{h_rejected} rejected for colliding with the 800k pool "
          f"({time.time()-t1:.1f}s)")

    os.rmdir(work_dir) if os.path.isdir(work_dir) and not os.listdir(work_dir) else None

    manifest = {
        "per_method_target": per_method,
        "methods": METHODS,
        "final_dataset_count": final_count,
        "final_dataset_path": final_path,
        "holdout_target": holdout_count,
        "holdout_collected": h_collected,
        "holdout_collisions_rejected": h_rejected,
        "holdout_path": holdout_path,
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(final_path.replace(".jsonl", "_manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-method", type=int, default=500,
                         help="boards per method (x4 methods = total dataset size)")
    parser.add_argument("--holdout", type=int, default=125,
                         help="size of the guaranteed-disjoint random holdout set")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=str, default="./full_dataset.jsonl")
    parser.add_argument("--holdout-out", type=str, default="./holdout_dataset.jsonl")
    parser.add_argument("--min-occupancy", type=int, default=12,
                         help="minimum total grid cells covered by blocks (out of 36)")
    parser.add_argument("--max-occupancy", type=int, default=26,
                         help="maximum total grid cells covered by blocks (out of 36)")
    args = parser.parse_args()

    manifest = build_full_dataset(
        per_method=args.per_method,
        holdout_count=args.holdout,
        work_dir="./_full_shards",
        final_path=args.out,
        holdout_path=args.holdout_out,
        workers=args.workers,
        min_occupancy=args.min_occupancy,
        max_occupancy=args.max_occupancy,
    )
    print("\nDone. Manifest:")
    print(json.dumps(manifest, indent=2))
