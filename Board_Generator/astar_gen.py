"""
METHOD 3: solved -> scrambled via A* (targeted priority search).

A second deliberate generation-bias condition. The heuristic here is
"how far RED currently sits from the exit" (bounded, normalized), so A*'s
guidance specifically pulls RED away from the exit while searching for a
state near target_depth. This is a narrower, more specific kind of bias
than BFS's breadth-layered exploration -- worth remembering when comparing
results: an "A*-trained" effect may really be a "RED-displaced" effect.

Also doubles as the dataset-building CLI for this method (folded in from
the old Methods/build_astar_dataset.py): sharded parallel generation,
dedupe by board signature, subsample to the exact target count, manifest
written alongside the output file.
"""

import random
import json
import argparse
import multiprocessing as mp
import time
import os
import heapq

from puzzle_manager import GRID, make_solved_seed, state_key, apply_state, get_moves, board_to_dict, board_signature

METHOD = "astar"


def astar_solved_to_scrambled(target_depth=8, num_blocks=7, max_expansions=50000):
    seed_blocks = make_solved_seed(num_blocks)
    start_key = state_key(seed_blocks)

    def heuristic(key):
        blocks = apply_state(seed_blocks, key)
        red = next(b for b in blocks if b.is_red)
        dist_from_exit = (GRID - red.length) - red.col
        # BOUNDED heuristic: normalized to [-1, 0]. An unbounded version
        # (raw -dist_from_exit) can dominate the abs(g - target_depth) term
        # entirely, causing the search to chase maximally-scrambled states
        # regardless of target depth. Bounded, it can only act as a
        # tie-breaker between states already equally close to target depth.
        return -(dist_from_exit / max(GRID, 1))

    counter = 0
    start_h = heuristic(start_key)
    frontier = [(abs(0 - target_depth) + start_h, 0, counter, start_key)]
    came_g = {start_key: 0}
    parent = {start_key: None}
    best_key, best_g, best_score = start_key, 0, abs(0 - target_depth) + start_h
    expansions = 0

    while frontier and expansions < max_expansions:
        f, g, _, key = heapq.heappop(frontier)
        if g != came_g.get(key, -1):
            continue
        expansions += 1

        score = abs(g - target_depth)
        if score < best_score:
            best_score, best_key, best_g = score, key, g
            if score == 0:
                break

        cur = apply_state(seed_blocks, key)
        for (bid, r, c) in get_moves(cur):
            new_blocks = [bb.clone() for bb in cur]
            new_blocks[bid].row, new_blocks[bid].col = r, c
            nk = state_key(new_blocks)
            ng = g + 1
            if nk in came_g and came_g[nk] <= ng:
                continue
            came_g[nk] = ng
            parent[nk] = (key, (bid, r, c))
            h = heuristic(nk)
            nf = abs(ng - target_depth) + h
            counter += 1
            heapq.heappush(frontier, (nf, ng, counter, nk))

    scrambled = apply_state(seed_blocks, best_key)

    path = []
    k = best_key
    while parent[k] is not None:
        prev_k, move = parent[k]
        path.append({"block_id": move[0], "row": move[1], "col": move[2]})
        k = prev_k
    path.reverse()

    return scrambled, best_g, path


def astar_solved_to_scrambled_tolerant(target_depth=8, num_blocks=7, tolerance=1,
                                        max_retries=20, max_expansions=50000):
    """A*'s priority queue can lock onto a state below/above target_depth.
    Retry with fresh random seeds (new seed board + new search) until the
    result lands within `tolerance` moves of target_depth, or give up after
    max_retries and return the closest one found."""
    best = None
    best_gap = None
    for _ in range(max_retries):
        board, depth, path = astar_solved_to_scrambled(
            target_depth=target_depth, num_blocks=num_blocks, max_expansions=max_expansions)
        gap = abs(depth - target_depth)
        if best is None or gap < best_gap:
            best, best_gap = (board, depth, path), gap
        if gap <= tolerance:
            return board, depth, path
    return best


# ---------------------------------------------------------------------------
# Dataset-building CLI (folded in from Methods/build_astar_dataset.py). Uses
# the tolerant wrapper directly rather than going through puzzle_manager's
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
            board, depth, path = astar_solved_to_scrambled_tolerant(target_depth=target_depth, num_blocks=num_blocks)
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
        board, depth, path = astar_solved_to_scrambled(target_depth=8, num_blocks=7)
        print(f"A*-scrambled board: {depth} moves from solved, path length {len(path)}")
    else:
        build_dataset(
            total=args.total, workers=args.workers,
            work_dir=f"./_{METHOD}_shards", final_path=args.out,
            num_blocks=args.num_blocks, depth_range=(args.min_depth, args.max_depth)
        )
