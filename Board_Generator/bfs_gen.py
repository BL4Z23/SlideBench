"""
METHOD 2: solved -> scrambled via plain BFS (full reachable-state exploration).

This is a deliberate generation-bias condition, not a neutral one: BFS
expands EVERY reachable state at each depth layer before moving to the
next, so it systematically over-represents "central," highly-connected
regions of the state graph relative to a single-path random walk. Returns
the move PATH (solved -> ... -> scrambled) too, useful for RL imitation
learning / reward shaping.

Also doubles as the dataset-building CLI for this method (folded in from
the old Methods/build_bfs_dataset.py): sharded parallel generation, dedupe
by board signature, subsample to the exact target count, manifest written
alongside the output file.
"""

import random
import json
import argparse
import multiprocessing as mp
import time
import os
from collections import deque

from puzzle_manager import make_solved_seed, state_key, apply_state, get_moves, board_to_dict, board_signature

METHOD = "bfs"


def bfs_solved_to_scrambled(target_depth=8, num_blocks=7, max_states=200000):
    """Explore the move-graph breadth-first from a solved state and return
    a board found at (or close to) target_depth, plus the move path taken
    to reach it. Guaranteed solvable in exactly that many moves, since that
    is how it was discovered."""
    seed_blocks = make_solved_seed(num_blocks)
    start_key = state_key(seed_blocks)
    seen = {start_key: 0}
    parent = {start_key: None}  # key -> (prev_key, move)
    queue = deque([start_key])
    candidates_at_depth = []

    while queue:
        key = queue.popleft()
        dist = seen[key]
        if dist == target_depth:
            candidates_at_depth.append(key)
            continue
        if dist > target_depth or len(seen) > max_states:
            break
        cur = apply_state(seed_blocks, key)
        for (bid, r, c) in get_moves(cur):
            new_blocks = [bb.clone() for bb in cur]
            new_blocks[bid].row, new_blocks[bid].col = r, c
            nk = state_key(new_blocks)
            if nk not in seen:
                seen[nk] = dist + 1
                parent[nk] = (key, (bid, r, c))
                queue.append(nk)

    if not candidates_at_depth:
        max_d = max(seen.values())
        candidates_at_depth = [k for k, d in seen.items() if d == max_d]

    chosen_key = random.choice(candidates_at_depth)
    scrambled = apply_state(seed_blocks, chosen_key)
    actual_depth = seen[chosen_key]

    # reconstruct move path solved -> scrambled
    path = []
    k = chosen_key
    while parent[k] is not None:
        prev_k, move = parent[k]
        path.append({"block_id": move[0], "row": move[1], "col": move[2]})
        k = prev_k
    path.reverse()

    return scrambled, actual_depth, path


# ---------------------------------------------------------------------------
# Dataset-building CLI (folded in from Methods/build_bfs_dataset.py). Calls
# bfs_solved_to_scrambled directly rather than going through puzzle_manager's
# GENERATORS registry, since puzzle_manager builds that registry FROM this
# file -- going back through it here would be a circular import.
# ---------------------------------------------------------------------------

def _gen_scramble_worker(args):
    seed, quota, num_blocks, depth_range, out_path = args
    random.seed(seed)
    collected = 0
    tries = 0
    with open(out_path, "w") as f:
        while collected < quota:
            tries += 1
            target_depth = random.randint(*depth_range)
            board, depth, path = bfs_solved_to_scrambled(target_depth=target_depth, num_blocks=num_blocks)
            rec = {
                "method": METHOD,
                "target_depth": target_depth,
                "actual_depth": depth,
                "move_path": path,
                "board": board_to_dict(board),
            }
            f.write(json.dumps(rec) + "\n")
            collected += 1
    return out_path, collected, tries


def build_dataset(total, workers, work_dir, final_path, num_blocks=7, depth_range=(4, 16)):
    os.makedirs(work_dir, exist_ok=True)
    parent = os.path.dirname(os.path.abspath(final_path))
    if parent: os.makedirs(parent, exist_ok=True)
    t0 = time.time()

    print(f"Generating {METHOD.upper()} dataset ({total} boards) using {workers} workers...")

    per_worker = int((total * 1.1) // workers + 1)
    jobs = []
    for w in range(workers):
        out_path = os.path.join(work_dir, f"{METHOD}_shard{w}.jsonl")
        seed = random.randint(0, 2 ** 31)
        jobs.append((seed, per_worker, num_blocks, depth_range, out_path))

    shard_paths = []
    with mp.Pool(workers) as pool:
        for path, collected, tries in pool.imap_unordered(_gen_scramble_worker, jobs):
            shard_paths.append(path)

    # Merge, Dedupe, and Subsample
    seen = set()
    valid_lines = []
    for sp in shard_paths:
        with open(sp) as f:
            for line in f:
                rec = json.loads(line)
                sig = board_signature(rec["board"])
                if sig not in seen:
                    seen.add(sig)
                    valid_lines.append(line)
        os.remove(sp)

    final_lines = random.sample(valid_lines, min(total, len(valid_lines)))

    with open(final_path, "w") as out_f:
        for line in final_lines:
            out_f.write(line)

    if os.path.isdir(work_dir) and not os.listdir(work_dir):
        os.rmdir(work_dir)

    manifest = {
        "method": METHOD,
        "target": total,
        "final_dataset_count": len(final_lines),
        "final_dataset_path": final_path,
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(final_path.replace(".jsonl", "_manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Done! {len(final_lines)} boards saved to {final_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=str, default=f"./{METHOD}_dataset.jsonl")
    parser.add_argument("--num-blocks", type=int, default=7)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--demo", action="store_true",
                         help="run a single quick sample instead of a full dataset build")
    args = parser.parse_args()

    if args.demo:
        random.seed()
        board, depth, path = bfs_solved_to_scrambled(target_depth=8, num_blocks=7)
        print(f"BFS-scrambled board: {depth} moves from solved, path length {len(path)}")
    else:
        build_dataset(
            total=args.total, workers=args.workers,
            work_dir=f"./_{METHOD}_shards", final_path=args.out,
            num_blocks=args.num_blocks, depth_range=(args.min_depth, args.max_depth)
        )
