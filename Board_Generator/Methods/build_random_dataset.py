import random
import json
import argparse
import multiprocessing as mp
import time
import os

from puzzle_core import (
    random_board_by_occupancy, board_occupancy,
    board_to_dict, board_signature, solve_bfs
)


def _gen_random_worker(args):
    seed, quota, out_path, min_moves, max_moves, min_occupancy, max_occupancy = args
    random.seed(seed)
    collected = 0
    tries = 0
    with open(out_path, "w") as f:
        while collected < quota:
            tries += 1
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


def build_dataset(total, workers, work_dir, final_path,
                  min_moves=4, max_moves=30,
                  min_occupancy=12, max_occupancy=26):
    os.makedirs(work_dir, exist_ok=True)
    parent = os.path.dirname(os.path.abspath(final_path))
    if parent: os.makedirs(parent, exist_ok=True)
    t0 = time.time()

    print(f"Generating Random dataset ({total} boards) using {workers} workers...")

    per_worker = int((total * 1.1) // workers + 1)
    jobs = []
    for w in range(workers):
        out_path = os.path.join(work_dir, f"random_shard{w}.jsonl")
        seed = random.randint(0, 2 ** 31)
        jobs.append((seed, per_worker, out_path, min_moves, max_moves,
                     min_occupancy, max_occupancy))

    shard_paths = []
    with mp.Pool(workers) as pool:
        for path, collected, tries in pool.imap_unordered(_gen_random_worker, jobs):
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

        # Write Manifest
    manifest = {
        "method": "random",
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
    parser.add_argument("--out", type=str, default="./random_dataset.jsonl")
    parser.add_argument("--min-occupancy", type=int, default=12)
    parser.add_argument("--max-occupancy", type=int, default=26)
    parser.add_argument("--min-moves", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=30)
    args = parser.parse_args()

    build_dataset(
        total=args.total, workers=args.workers,
        work_dir="./_random_shards", final_path=args.out,
        min_moves=args.min_moves, max_moves=args.max_moves,
        min_occupancy=args.min_occupancy, max_occupancy=args.max_occupancy
    )