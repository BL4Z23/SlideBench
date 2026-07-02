import random
import json
import argparse
import time
import os

from puzzle_core import (
    random_board_by_occupancy, board_occupancy,
    board_to_dict, board_signature, solve_bfs
)

def build_holdout(count, exclude_files, out_path, min_moves=4, max_moves=30,
                  min_occupancy=12, max_occupancy=26):
    global_seen = set()
    for fpath in exclude_files:
        if not os.path.exists(fpath):
            print(f"Skipping {fpath} (not found)")
            continue
        print(f"Loading signatures from {fpath}...")
        with open(fpath, 'r') as f:
            for line in f:
                rec = json.loads(line)
                global_seen.add(board_signature(rec["board"]))
                
    print(f"Loaded {len(global_seen)} total signatures to exclude.")
    
    collected = 0
    tries = 0
    collisions_rejected = 0
    local_seen = set() # Dedupe within the holdout itself
    
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent: os.makedirs(parent, exist_ok=True)
        
    t0 = time.time()
    with open(out_path, "w") as f:
        while collected < count:
            tries += 1
            blocks = random_board_by_occupancy(
                min_occupancy=min_occupancy, max_occupancy=max_occupancy)
            if blocks is None or len(blocks) < 3:
                continue
            
            board = board_to_dict(blocks)
            board["occupancy"] = board_occupancy(blocks)
            sig = board_signature(board)

            # Core Logic: Reject if it collides with the main datasets
            if sig in global_seen:
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

            if collected % 5000 == 0:
                print(f"  ...collected {collected}/{count}")

    # Write Manifest
    manifest = {
        "method": "random_holdout",
        "target": count,
        "final_dataset_count": collected,
        "collisions_rejected": collisions_rejected,
        "final_dataset_path": out_path,
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(out_path.replace(".jsonl", "_manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Done! {collected} boards saved. {collisions_rejected} collisions prevented. Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=50000)
    parser.add_argument("--exclude", nargs='+', 
                        default=["random_dataset.jsonl", "bfs_dataset.jsonl", "astar_dataset.jsonl", "idastar_dataset.jsonl"], 
                        help="List of dataset files to read and exclude")
    parser.add_argument("--out", type=str, default="./holdout_dataset.jsonl")
    parser.add_argument("--min-occupancy", type=int, default=12)
    parser.add_argument("--max-occupancy", type=int, default=26)
    args = parser.parse_args()

    build_holdout(
        count=args.total,
        exclude_files=args.exclude,
        out_path=args.out,
        min_occupancy=args.min_occupancy,
        max_occupancy=args.max_occupancy
    )