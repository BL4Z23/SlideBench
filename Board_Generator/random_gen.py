"""
METHOD 1: random placement + BFS solvability validation.

Boards are built by dropping blocks at random positions (not by scrambling
backward from a solved state). A solver is used ONLY as an accept/reject
filter after the fact -- it never steers which geometry gets chosen, so
this is the lightest-touch of the four methods, closest to a neutral
baseline (though not perfectly zero-touch, since some random geometries
are more likely to survive the solvability filter than others).

Also doubles as the dataset-building CLI for this method (folded in from
the old Methods/build_random_dataset.py): sharded parallel generation,
dedupe by board signature, subsample to the exact target count, manifest
written alongside the output file. This is separate from
generate_puzzles_parallel() below, which returns results in-memory rather
than writing sharded files to disk -- keep both, they serve different uses
(quick in-process sampling vs. a full production dataset build).
"""

import random
import json
import argparse
import multiprocessing as mp
import time
import os

from puzzle_manager import (
    GRID, EXIT_ROW, Block, try_place_block, occupied_cells, in_bounds,
    solve_bfs, board_to_dict, board_signature, dedupe_records, count_shortest_solutions,
)


def random_board(num_blocks=6):
    """Randomly drop the red block + N other blocks of random size/orientation."""
    blocks = []
    red_len = 2  # red block is ALWAYS length 2 in the reference game, never 3
    red = try_place_block(blocks, red_len, horizontal=True, is_red=True)
    if red is None:
        return None
    blocks.append(red)

    attempts = 0
    while len(blocks) - 1 < num_blocks and attempts < 500:
        attempts += 1
        length = random.choice([2, 2, 3])  # bias toward 2-length blocks
        horizontal = random.choice([True, False])
        b = try_place_block(blocks, length, horizontal)
        if b:
            b.id = len(blocks)
            blocks.append(b)
    return blocks


def board_occupancy(blocks):
    """Total number of grid cells covered by all blocks combined."""
    return sum(b.length for b in blocks)


def random_board_by_occupancy(min_occupancy=12, max_occupancy=26,
                               piece_lengths=(2, 3), length_weights=(0.65, 0.35),
                               max_blocks=12, max_attempts=500,
                               red_row_avoid_prob=0.0):
    """Randomly drop the red block, then keep adding non-red blocks (each of
    length 2 or 3, per `piece_lengths`/`length_weights`) until total occupied
    cells lands inside [min_occupancy, max_occupancy].

    `red_row_avoid_prob` biases placement toward a more FAVORABLE starting
    position: with this probability, a candidate placement that would occupy
    any cell in the red block's row (EXIT_ROW) is rejected and re-rolled
    instead. Higher values -> more likely the red block's row stays clear
    near the start, which is what makes "easy" boards feel easy. Use 0.0
    (no bias) for hard/very-hard tiers and 0.6-0.85 for easy tiers.

    Returns None if it can't reach at least min_occupancy within the attempt
    budget, or if red placement fails.
    """
    grid_cells = GRID * GRID
    max_occupancy = min(max_occupancy, grid_cells)
    if min_occupancy > max_occupancy:
        raise ValueError("min_occupancy cannot exceed max_occupancy")

    blocks = []
    red_len = 2
    red = try_place_block(blocks, red_len, horizontal=True, is_red=True)
    if red is None:
        return None
    blocks.append(red)
    occ = board_occupancy(blocks)

    attempts = 0
    while occ < max_occupancy and len(blocks) < max_blocks and attempts < max_attempts:
        attempts += 1
        room_left = max_occupancy - occ
        candidates = [ln for ln in piece_lengths if ln <= room_left]
        if not candidates:
            break
        weights = [length_weights[piece_lengths.index(ln)] for ln in candidates]
        length = random.choices(candidates, weights=weights, k=1)[0]
        horizontal = random.choice([True, False])
        b = try_place_block(blocks, length, horizontal)
        if b is None:
            continue

        if red_row_avoid_prob > 0:
            hits_red_row = any(r == EXIT_ROW for r, c in b.cells())
            if hits_red_row and random.random() < red_row_avoid_prob:
                continue  # re-roll: keep the escape lane clearer for easy boards

        b.id = len(blocks)
        blocks.append(b)
        occ += length

    if occ < min_occupancy:
        return None  # couldn't pack enough pieces to hit the floor -- discard
    return blocks


def generate_valid_puzzles(n, min_moves=4, max_moves=25, num_blocks_range=(5, 8)):
    """Keep generating random boards until we have n valid ones."""
    results = []
    tries = 0
    while len(results) < n:
        tries += 1
        nb = random.randint(*num_blocks_range)
        blocks = random_board(num_blocks=nb)
        if blocks is None or len(blocks) < 3:
            continue
        moves = solve_bfs(blocks)
        if moves is None:
            continue  # unsolvable, discard
        if moves < min_moves or moves > max_moves:
            continue  # too trivial or too brutal, discard
        results.append({"difficulty_moves": moves, "board": board_to_dict(blocks)})
    return results, tries


def _worker_generate(args):
    seed, target, min_moves, max_moves, num_blocks_range, require_unique = args
    random.seed(seed)
    out = []
    tries = 0
    while len(out) < target:
        tries += 1
        nb = random.randint(*num_blocks_range)
        blocks = random_board(num_blocks=nb)
        if blocks is None or len(blocks) < 3:
            continue
        if require_unique:
            moves, n_paths = count_shortest_solutions(blocks)
            if moves is None or n_paths != 1:
                continue
        else:
            moves = solve_bfs(blocks)
            if moves is None:
                continue
        if moves < min_moves or moves > max_moves:
            continue
        out.append({"difficulty_moves": moves, "board": board_to_dict(blocks)})
    return out, tries


def generate_puzzles_parallel(total, workers=8, min_moves=4, max_moves=25,
                               num_blocks_range=(5, 8), require_unique=False):
    """In-memory parallel generation -- returns results directly rather than
    writing sharded files to disk. Good for quick sampling / use inside other
    functions (e.g. puzzle_manager's difficulty-matched sets). For a full
    production dataset build with sharding + manifest, use build_dataset()
    below instead."""
    per_worker = total // workers + 1
    jobs = [
        (random.randint(0, 2**31), per_worker, min_moves, max_moves,
         num_blocks_range, require_unique)
        for _ in range(workers)
    ]
    results = []
    total_tries = 0
    with mp.Pool(workers) as pool:
        for out, tries in pool.imap_unordered(_worker_generate, jobs):
            results.extend(out)
            total_tries += tries
    results = dedupe_records(results)
    return results[:total], total_tries


# ---------------------------------------------------------------------------
# Dataset-building CLI (folded in from Methods/build_random_dataset.py).
# Unlike generate_puzzles_parallel() above, this writes sharded JSONL files
# to disk, dedupes + subsamples across shards, and writes a manifest -- the
# production path for building a full-scale dataset.
# ---------------------------------------------------------------------------

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
    parser.add_argument("--demo", action="store_true",
                         help="run quick in-process samples instead of a full dataset build")
    args = parser.parse_args()

    if args.demo:
        random.seed()
        print("=== Method 1: random placement + BFS validation (single-process) ===")
        puzzles, tries = generate_valid_puzzles(10)
        print(f"Generated {len(puzzles)} valid puzzles out of {tries} attempts "
              f"({len(puzzles)/tries:.1%} acceptance)")

        print("\n=== Method 1b: same, parallelized across processes ===")
        t0 = time.time()
        par_puzzles, par_tries = generate_puzzles_parallel(40, workers=8)
        print(f"Generated {len(par_puzzles)} puzzles from {par_tries} attempts "
              f"in {time.time()-t0:.2f}s using 8 workers")

        print("\n=== Method 1c: same, requiring a UNIQUE shortest solution ===")
        t0 = time.time()
        uniq_puzzles, uniq_tries = generate_puzzles_parallel(
            10, workers=8, require_unique=True, min_moves=3, max_moves=20)
        print(f"Generated {len(uniq_puzzles)} unique-solution puzzles from "
              f"{uniq_tries} attempts in {time.time()-t0:.2f}s")
    else:
        build_dataset(
            total=args.total, workers=args.workers,
            work_dir="./_random_shards", final_path=args.out,
            min_moves=args.min_moves, max_moves=args.max_moves,
            min_occupancy=args.min_occupancy, max_occupancy=args.max_occupancy
        )
