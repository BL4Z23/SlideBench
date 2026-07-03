"""
METHOD 4: solved -> scrambled via IDA* (iterative deepening A*).

A third distinct deliberate generation-bias condition. IDA* repeatedly does
depth-limited DFS, raising the f-bound each round (rather than keeping a
priority queue like A*, or expanding whole layers like BFS). Its search
order is depth-first within each bound, so it tends to commit early to one
block's moves before backtracking -- structurally different from both
BFS's breadth-first layering and A*'s best-first ordering.

Also doubles as the dataset-building CLI for this method (folded in from
the old Methods/build_idastar_dataset.py): sharded parallel generation,
dedupe by board signature, subsample to the exact target count, manifest
written alongside the output file.
"""

import random
import json
import argparse
import multiprocessing as mp
import time
import os

from puzzle_manager import GRID, make_solved_seed, state_key, apply_state, get_moves, board_to_dict, board_signature

METHOD = "idastar"


def ida_star_solved_to_scrambled(target_depth=8, num_blocks=7, max_node_visits=300000):
    seed_blocks = make_solved_seed(num_blocks)
    start_key = state_key(seed_blocks)

    def heuristic(key):
        blocks = apply_state(seed_blocks, key)
        red = next(b for b in blocks if b.is_red)
        dist_from_exit = (GRID - red.length) - red.col
        return -(dist_from_exit / max(GRID, 1))  # same bounded fix as A*

    best = {"key": start_key, "g": 0, "score": abs(0 - target_depth) + heuristic(start_key)}
    best_path = {"path": []}
    visits = [0]

    def dfs(key, g, bound, path):
        visits[0] += 1
        if visits[0] > max_node_visits:
            return None

        h = heuristic(key)
        f = g + h
        score = abs(g - target_depth)
        if score < best["score"]:
            best["score"] = score
            best["key"] = key
            best["g"] = g
            best_path["path"] = list(path)
            if score == 0:
                return "FOUND"

        if g >= target_depth or f > bound:
            return f  # report minimal f exceeding bound, for next iteration

        cur = apply_state(seed_blocks, key)
        min_exceed = None
        for (bid, r, c) in get_moves(cur):
            new_blocks = [bb.clone() for bb in cur]
            new_blocks[bid].row, new_blocks[bid].col = r, c
            nk = state_key(new_blocks)
            path.append({"block_id": bid, "row": r, "col": c})
            result = dfs(nk, g + 1, bound, path)
            path.pop()
            if result == "FOUND":
                return "FOUND"
            if result is not None and (min_exceed is None or result < min_exceed):
                min_exceed = result
            if visits[0] > max_node_visits:
                return None
        return min_exceed

    bound = heuristic(start_key)
    for _ in range(64):  # cap iterations so it can't loop forever
        result = dfs(start_key, 0, bound, [])
        if result == "FOUND" or result is None:
            break
        bound = result

    scrambled = apply_state(seed_blocks, best["key"])
    return scrambled, best["g"], best_path["path"]


# ---------------------------------------------------------------------------
# Dataset-building CLI (folded in from Methods/build_idastar_dataset.py).
# Calls ida_star_solved_to_scrambled directly rather than going through
# puzzle_manager's GENERATORS registry, since puzzle_manager builds that
# registry FROM this file -- going back through it here would be a
# circular import.
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
            board, depth, path = ida_star_solved_to_scrambled(target_depth=target_depth, num_blocks=num_blocks)
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
        board, depth, path = ida_star_solved_to_scrambled(target_depth=8, num_blocks=7)
        print(f"IDA*-scrambled board: {depth} moves from solved, path length {len(path)}")
    else:
        build_dataset(
            total=args.total, workers=args.workers,
            work_dir=f"./_{METHOD}_shards", final_path=args.out,
            num_blocks=args.num_blocks, depth_range=(args.min_depth, args.max_depth)
        )
