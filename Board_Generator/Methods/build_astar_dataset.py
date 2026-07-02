import random
import json
import argparse
import multiprocessing as mp
import time
import os

from puzzle_core import GENERATORS, board_to_dict, board_signature

METHOD = "astar"


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
        jobs.append((METHOD, seed, per_worker, num_blocks, depth_range, out_path))

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
    args = parser.parse_args()

    build_dataset(
        total=args.total, workers=args.workers,
        work_dir=f"./_{METHOD}_shards", final_path=args.out,
        num_blocks=args.num_blocks, depth_range=(args.min_depth, args.max_depth)
    )