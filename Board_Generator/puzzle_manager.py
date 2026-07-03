"""
Shared engine for the sliding-block puzzle project.

This module owns everything that every generation method (random / bfs /
astar / idastar) and the difficulty module both depend on:
  - the Block model and board geometry primitives (occupancy, legal moves)
  - the ground-truth BFS solver (bridged to the fast bitmask implementation)
  - board (de)serialization + deduplication signatures
  - the GENERATORS registry, and cross-method dataset-building helpers

Difficulty scoring lives in difficulty.py, not here -- see that module for
why (it's solver-independent by design and shouldn't be coupled to any one
generation method). board_to_dict() below needs difficulty scores, and
difficulty.py needs occupied_cells()/GRID from here, so the import between
them is deferred (done inside the function body) to avoid a circular
top-level import, exactly the same pattern already used for solve_bfs()'s
dependency on bitmask_solver.

IMPORT ORDER NOTE: the GENERATORS registry and the cross-method helpers
near the bottom of this file import from random_gen.py / bfs_gen.py /
astar_gen.py / idastar_gen.py, and those four modules import primitives
(Block, state_key, occupied_cells, solve_bfs, etc.) from THIS module. That
works because those imports are placed at the bottom of this file, after
every primitive they need is already defined -- by the time Python reaches
those lines, everything the four generator modules ask for already exists
in this module's namespace. Don't move primitive definitions below that
point, and don't move the bottom imports above their dependencies.
"""

import random
import json
from collections import deque

GRID = 6
EXIT_ROW = 2  # row index (0-based) the red block must clear


class Block:
    __slots__ = ("id", "row", "col", "length", "horizontal", "is_red")

    def __init__(self, id, row, col, length, horizontal, is_red=False):
        self.id = id
        self.row = row
        self.col = col
        self.length = length
        self.horizontal = horizontal
        self.is_red = is_red

    def cells(self, row=None, col=None):
        r = self.row if row is None else row
        c = self.col if col is None else col
        if self.horizontal:
            return [(r, c + i) for i in range(self.length)]
        else:
            return [(r + i, c) for i in range(self.length)]

    def clone(self):
        return Block(self.id, self.row, self.col, self.length, self.horizontal, self.is_red)


def occupied_cells(blocks):
    occ = {}
    for b in blocks:
        for cell in b.cells():
            occ[cell] = b.id
    return occ


def in_bounds(cells):
    return all(0 <= r < GRID and 0 <= c < GRID for r, c in cells)


def try_place_block(blocks, length, horizontal, is_red=False, max_tries=200):
    occ = occupied_cells(blocks)
    for _ in range(max_tries):
        if is_red:
            row = EXIT_ROW
            col = random.randint(0, GRID - length)
        else:
            row = random.randint(0, GRID - (length if not horizontal else 1))
            col = random.randint(0, GRID - (length if horizontal else 1))
        b = Block(len(blocks), row, col, length, horizontal, is_red)
        cells = b.cells()
        if not in_bounds(cells):
            continue
        if any(cell in occ for cell in cells):
            continue
        return b
    return None


def state_key(blocks):
    return tuple((b.row, b.col) for b in blocks)


def apply_state(blocks, key):
    new_blocks = []
    for b, (r, c) in zip(blocks, key):
        nb = b.clone()
        nb.row, nb.col = r, c
        new_blocks.append(nb)
    return new_blocks


def get_moves(blocks):
    """Return list of (block_id, new_row, new_col) reachable in one slide
    (any distance, not just one cell)."""
    occ = occupied_cells(blocks)
    moves = []
    for b in blocks:
        for cell in b.cells():
            del occ[cell]  # temporarily remove this block from occ map
        if b.horizontal:
            c = b.col - 1
            while c >= 0 and (b.row, c) not in occ:
                moves.append((b.id, b.row, c))
                c -= 1
            c = b.col + 1
            max_c = GRID - b.length
            while c <= max_c and (b.row, c + b.length - 1) not in occ:
                moves.append((b.id, b.row, c))
                c += 1
        else:
            r = b.row - 1
            while r >= 0 and (r, b.col) not in occ:
                moves.append((b.id, r, b.col))
                r -= 1
            r = b.row + 1
            max_r = GRID - b.length
            while r <= max_r and (r + b.length - 1, b.col) not in occ:
                moves.append((b.id, r, b.col))
                r += 1
        for cell in b.cells():
            occ[cell] = b.id
    return moves


def is_solved(blocks):
    red = next(b for b in blocks if b.is_red)
    return red.col + red.length - 1 == GRID - 1


def solve_bfs(blocks, max_states=200000):
    """BFS over block configurations. Returns min #moves to solve, or None.
    Internally delegates to the bitmask-based fast_solver (3x+ faster, same
    result) -- kept under this name so every caller gets the speedup
    automatically with zero changes on their end."""
    from bitmask_solver import blocks_to_fast, solve_bfs_fast
    geom, state, red_idx = blocks_to_fast(blocks)
    return solve_bfs_fast(geom, state, red_idx, max_states=max_states)


def _solve_bfs_reference(blocks, max_states=200000):
    """Original dict/object-based implementation, kept for reference and
    correctness testing against the fast version."""
    start_key = state_key(blocks)
    if is_solved(blocks):
        return 0
    seen = {start_key}
    queue = deque([(start_key, 0)])
    template = blocks

    while queue:
        key, dist = queue.popleft()
        cur = apply_state(template, key)
        for (bid, r, c) in get_moves(cur):
            new_blocks = [bb.clone() for bb in cur]
            new_blocks[bid].row, new_blocks[bid].col = r, c
            nk = state_key(new_blocks)
            if nk in seen:
                continue
            if is_solved(new_blocks):
                return dist + 1
            seen.add(nk)
            queue.append((nk, dist + 1))
            if len(seen) > max_states:
                return None
    return None


def board_to_dict(blocks):
    # Deferred import: difficulty.py imports occupied_cells/GRID from this
    # module, so importing difficulty at the TOP of this file would be
    # circular. By the time this function is actually CALLED (not just
    # defined), both modules have finished loading, so this is safe.
    from difficulty import compute_difficulty_score
    diff = compute_difficulty_score(blocks)
    return {
        "blocks": [
            {"id": b.id, "row": b.row, "col": b.col, "length": b.length,
             "horizontal": b.horizontal, "is_red": b.is_red}
            for b in blocks
        ],
        "grid": GRID,
        "exit_row": EXIT_ROW,
        **diff,
    }


def board_signature(board_dict):
    """Canonical, order-independent fingerprint of a board's geometry, for
    deduplication. Block `id` is generation-order-dependent and not part of
    the signature -- two boards with identical block shapes/positions but
    different placement order are the same puzzle and should dedupe."""
    return frozenset(
        (b["row"], b["col"], b["length"], b["horizontal"], b["is_red"])
        for b in board_dict["blocks"]
    )


def dedupe_records(records, board_key=lambda r: r["board"]):
    """Drop records whose underlying board geometry is a duplicate of one
    already kept. Keeps the first occurrence, preserves input order."""
    seen = set()
    out = []
    for r in records:
        sig = board_signature(board_key(r))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
    return out


def count_shortest_solutions(blocks, max_states=300000):
    """BFS but track HOW MANY distinct shortest paths reach a solved state.
    Used to filter for puzzles with a (close to) unique solution."""
    start_key = state_key(blocks)
    if is_solved(blocks):
        return 0, 1
    seen = {start_key: 0}
    queue = deque([start_key])
    template = blocks
    best_dist = None
    solved_paths = 0

    while queue:
        key = queue.popleft()
        dist = seen[key]
        if best_dist is not None and dist >= best_dist:
            continue
        cur = apply_state(template, key)
        for (bid, r, c) in get_moves(cur):
            new_blocks = [bb.clone() for bb in cur]
            new_blocks[bid].row, new_blocks[bid].col = r, c
            nk = state_key(new_blocks)
            nd = dist + 1
            if is_solved(new_blocks):
                if best_dist is None:
                    best_dist = nd
                if nd == best_dist:
                    solved_paths += 1
                continue
            if nk not in seen:
                seen[nk] = nd
                queue.append(nk)
                if len(seen) > max_states:
                    return best_dist, (solved_paths if best_dist else 0)
    return best_dist, solved_paths


def make_solved_seed(num_blocks=7):
    """Builds a solved board: red touching the exit, then randomly drops
    the rest of the roster. Shared starting point for bfs_gen/astar_gen/
    idastar_gen's reverse-search generation."""
    blocks = []
    red_len = 2  # red block is ALWAYS length 2 in the reference game, never 3
    red = Block(0, EXIT_ROW, GRID - red_len, red_len, True, is_red=True)
    blocks.append(red)

    attempts = 0
    while len(blocks) - 1 < num_blocks and attempts < 500:
        attempts += 1
        length = random.choice([2, 2, 3])
        horizontal = random.choice([True, False])
        b = try_place_block(blocks, length, horizontal)
        if b:
            b.id = len(blocks)
            blocks.append(b)
    return blocks


# ---------------------------------------------------------------------------
# Bottom-of-file assembly: import each generation method's public function
# only AFTER every primitive above is fully defined (see the IMPORT ORDER
# NOTE at the top of this file for why this ordering matters).
# ---------------------------------------------------------------------------

from random_gen import (
    random_board, random_board_by_occupancy, board_occupancy,
    generate_valid_puzzles, generate_puzzles_parallel,
)
from bfs_gen import bfs_solved_to_scrambled
from astar_gen import astar_solved_to_scrambled, astar_solved_to_scrambled_tolerant
from idastar_gen import ida_star_solved_to_scrambled

GENERATORS = {
    "bfs": bfs_solved_to_scrambled,
    "astar": astar_solved_to_scrambled_tolerant,
    "idastar": ida_star_solved_to_scrambled,
}


def generate_training_set(method, count, depth_range=(4, 16), num_blocks=7):
    """Generate `count` boards for one method, evenly spread across depths."""
    assert method in GENERATORS, f"unknown method {method}, pick from {list(GENERATORS)}"
    gen_fn = GENERATORS[method]
    lo, hi = depth_range
    depths = [lo + (i * (hi - lo)) // max(count - 1, 1) for i in range(count)]
    random.shuffle(depths)

    out = []
    for target_depth in depths:
        board, actual_depth, path = gen_fn(target_depth=target_depth, num_blocks=num_blocks)
        out.append({
            "method": method,
            "target_depth": target_depth,
            "actual_depth": actual_depth,
            "move_path": path,
            "board": board_to_dict(board),
        })
    return out


def generate_matched_training_sets(count_per_method=50, depth_range=(4, 16), num_blocks=7,
                                    methods=("bfs", "astar", "idastar")):
    """Build difficulty-matched training sets for each generation method, so
    you can train separate agents (bfs-trained / astar-trained / idastar-trained
    / random-trained) and compare them on a shared neutral (Method 1) eval set."""
    datasets = {}
    for m in methods:
        datasets[m] = generate_training_set(m, count_per_method, depth_range, num_blocks)
    return datasets


# ---------------------------------------------------------------------------
# CROSS-METHOD DIFFICULTY MATCHING
# ---------------------------------------------------------------------------
# The depth-stratified sets above match on SEARCH DEPTH, which is itself a
# solver-defined notion (different for BFS/A*/IDA*, and undefined for random
# Method 1 boards until you solve them). To compare apples-to-apples across
# ALL FOUR methods -- including plain random generation -- bin every board
# by the solver-agnostic `difficulty_score` (from difficulty.py) instead,
# then oversample each method and subsample down to equal counts per bin.
# This way "difficulty" means the same thing across all four datasets, and
# none of the four generation processes gets to define what "hard" means
# for the others.

def _bin_index(score, bin_edges):
    for i, edge in enumerate(bin_edges):
        if score < edge:
            return i
    return len(bin_edges)


def generate_difficulty_matched_sets(target_per_bin=20, n_bins=5, num_blocks=7,
                                      methods=("random", "bfs", "astar", "idastar"),
                                      oversample_factor=4, depth_range=(4, 16)):
    """Produce N boards per difficulty bin, per method, all binned on the
    same solver-agnostic difficulty_score -- so dataset sizes AND difficulty
    distributions are matched across random/bfs/astar/idastar."""
    pools = {m: [] for m in methods}

    # 1. oversample each method generously
    for m in methods:
        want = target_per_bin * n_bins * oversample_factor // 4
        if m == "random":
            boards, _ = generate_puzzles_parallel(
                want, workers=8, min_moves=1, max_moves=999,
                num_blocks_range=(num_blocks - 1, num_blocks + 1))
            pools[m] = [b["board"] for b in boards]
        else:
            for _ in range(want):
                target_depth = random.randint(*depth_range)
                board, depth, path = GENERATORS[m](target_depth=target_depth, num_blocks=num_blocks)
                pools[m].append(board_to_dict(board))

    # 2. establish shared bin edges from the pooled difficulty_score values
    all_scores = sorted(b["difficulty_score"] for pool in pools.values() for b in pool)
    if not all_scores:
        return {m: [] for m in methods}
    step = max(len(all_scores) // n_bins, 1)
    bin_edges = [all_scores[min(i * step, len(all_scores) - 1)] for i in range(1, n_bins)]

    # 3. bucket each method's pool into the shared bins
    binned = {m: {i: [] for i in range(n_bins)} for m in methods}
    for m in methods:
        for b in pools[m]:
            idx = _bin_index(b["difficulty_score"], bin_edges)
            binned[m][idx].append(b)

    # 4. subsample every method down to the SAME count per bin (the min
    # available across methods for that bin, capped at target_per_bin) so
    # all four datasets end up equal-sized and equal-difficulty-distributed
    final = {m: [] for m in methods}
    for i in range(n_bins):
        available = min(len(binned[m][i]) for m in methods)
        take = min(target_per_bin, available)
        if take == 0:
            continue
        for m in methods:
            sample = random.sample(binned[m][i], take)
            final[m].extend(sample)

    return final


if __name__ == "__main__":
    random.seed()

    print("=== Method 1: random placement + BFS validation (single-process sample) ===")
    puzzles, tries = generate_valid_puzzles(10)
    print(f"Generated {len(puzzles)} valid puzzles out of {tries} attempts "
          f"({len(puzzles)/tries:.1%} acceptance)")

    print("\n=== Method 2: solved -> scrambled via BFS (with move path) ===")
    board, depth, path = bfs_solved_to_scrambled(target_depth=8, num_blocks=7)
    print(f"BFS-scrambled board: {depth} moves from solved, path length {len(path)}")

    print("\n=== Method 3: solved -> scrambled via A* (with move path) ===")
    board2, depth2, path2 = astar_solved_to_scrambled(target_depth=8, num_blocks=7)
    print(f"A*-scrambled board: {depth2} moves from solved, path length {len(path2)}")

    print("\n=== Method 4: solved -> scrambled via IDA* (with move path) ===")
    board3, depth3, path3 = ida_star_solved_to_scrambled(target_depth=8, num_blocks=7)
    print(f"IDA*-scrambled board: {depth3} moves from solved, path length {len(path3)}")

    print("\n=== Depth-stratified, difficulty-matched training sets (bfs/astar/idastar) ===")
    import time
    t0 = time.time()
    matched = generate_matched_training_sets(count_per_method=15, depth_range=(4, 14))
    for m, boards in matched.items():
        depths_actual = [b["actual_depth"] for b in boards]
        print(f"  {m:10s}: {len(boards)} boards, depth range "
              f"{min(depths_actual)}-{max(depths_actual)} "
              f"in {time.time()-t0:.2f}s (cumulative)")

    print("\n=== Difficulty-matched sets across ALL FOUR methods (solver-agnostic binning) ===")
    t0 = time.time()
    matched4 = generate_difficulty_matched_sets(target_per_bin=5, n_bins=4, num_blocks=7)
    for m, boards in matched4.items():
        scores = [b["difficulty_score"] for b in boards]
        if scores:
            print(f"  {m:8s}: n={len(boards):3d}  score range "
                  f"{min(scores):.1f}-{max(scores):.1f}  avg={sum(scores)/len(scores):.1f}")
        else:
            print(f"  {m:8s}: n=0 (no boards survived binning)")
    print(f"  (took {time.time()-t0:.1f}s)")
