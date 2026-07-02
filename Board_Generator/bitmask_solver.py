"""
Fast bitmask-based solver, drop-in alternative to solve_bfs() in
generate_puzzles.py. Same algorithm (BFS over the state graph), but:

  - Board occupancy represented as a single 36-bit Python int instead of a
    dict of (row, col) -> block_id. Overlap checks become bitwise AND
    instead of dict lookups.
  - Block geometry (length, orientation) is STATIC per block index and
    precomputed once; only (row, col) varies per state, so state keys are
    compact tuples of ints, hashing faster than the dict-clone approach.
  - Cloning a state is now just rebuilding a small tuple, not deep-cloning
    Block objects with __slots__ attribute copies.

This is still pure CPU work (not GPU-able, as discussed), but cuts the
constant-factor overhead of the dict/object-based version substantially --
on a multi-core CPU like a 9700X, this compounds nicely with raising
worker count, since each worker now burns less time per board.
"""

import random
from collections import deque

GRID = 6
EXIT_ROW = 2


def precompute_cell_masks():
    """Bitmask for every individual cell (row, col) -> 1 << index."""
    masks = {}
    for r in range(GRID):
        for c in range(GRID):
            masks[(r, c)] = 1 << (r * GRID + c)
    return masks

CELL_MASK = precompute_cell_masks()


def block_mask(row, col, length, horizontal):
    """Bitmask of all cells this block occupies at (row, col)."""
    m = 0
    if horizontal:
        for i in range(length):
            m |= CELL_MASK[(row, col + i)]
    else:
        for i in range(length):
            m |= CELL_MASK[(row + i, col)]
    return m


def occupancy_from_state(geom, state):
    """geom: tuple of (length, horizontal) per block index, static.
    state: tuple of (row, col) per block index, varies.
    Returns the combined occupancy bitmask."""
    occ = 0
    for (length, horizontal), (row, col) in zip(geom, state):
        occ |= block_mask(row, col, length, horizontal)
    return occ


def get_moves_fast(geom, state, occ):
    """Yield (block_idx, new_row, new_col) for every legal one-slide move,
    using bitmask overlap checks instead of dict lookups."""
    moves = []
    for idx, ((length, horizontal), (row, col)) in enumerate(zip(geom, state)):
        my_mask = block_mask(row, col, length, horizontal)
        occ_others = occ & ~my_mask  # remove this block's own cells

        if horizontal:
            c = col - 1
            while c >= 0:
                cell_m = CELL_MASK[(row, c)]
                if occ_others & cell_m:
                    break
                moves.append((idx, row, c))
                c -= 1
            c = col + 1
            max_c = GRID - length
            while c <= max_c:
                # need the trailing edge cell clear
                edge_m = CELL_MASK[(row, c + length - 1)]
                if occ_others & edge_m:
                    break
                moves.append((idx, row, c))
                c += 1
        else:
            r = row - 1
            while r >= 0:
                cell_m = CELL_MASK[(r, col)]
                if occ_others & cell_m:
                    break
                moves.append((idx, r, col))
                r -= 1
            r = row + 1
            max_r = GRID - length
            while r <= max_r:
                edge_m = CELL_MASK[(r + length - 1, col)]
                if occ_others & edge_m:
                    break
                moves.append((idx, r, col))
                r += 1
    return moves


def is_solved_fast(geom, state, red_idx):
    length, horizontal = geom[red_idx]
    row, col = state[red_idx]
    return col + length - 1 == GRID - 1


def solve_bfs_fast(geom, start_state, red_idx, max_states=200000):
    """Same semantics as solve_bfs() in generate_puzzles.py: returns min
    moves to solve, or None if unsolvable within max_states."""
    if is_solved_fast(geom, start_state, red_idx):
        return 0

    seen = {start_state}
    queue = deque([(start_state, 0)])

    while queue:
        state, dist = queue.popleft()
        occ = occupancy_from_state(geom, state)
        for (idx, r, c) in get_moves_fast(geom, state, occ):
            new_state = list(state)
            new_state[idx] = (r, c)
            new_state = tuple(new_state)
            if new_state in seen:
                continue
            if is_solved_fast(geom, new_state, red_idx):
                return dist + 1
            seen.add(new_state)
            queue.append((new_state, dist + 1))
            if len(seen) > max_states:
                return None
    return None


# ---------------------------------------------------------------------------
# Bridge: convert a Block-list board (from puzzle_core.py) into the
# (geom, state, red_idx) representation this module uses.
# ---------------------------------------------------------------------------

def blocks_to_fast(blocks):
    geom = tuple((b.length, b.horizontal) for b in blocks)
    state = tuple((b.row, b.col) for b in blocks)
    red_idx = next(i for i, b in enumerate(blocks) if b.is_red)
    return geom, state, red_idx


if __name__ == "__main__":
    import time
    from puzzle_core import random_board, solve_bfs

    random.seed(7)
    boards = []
    while len(boards) < 60:
        nb = random.randint(5, 9)
        b = random_board(num_blocks=nb)
        if b:
            boards.append(b)

    print(f"Benchmarking on {len(boards)} random boards...")

    t0 = time.time()
    old_results = [solve_bfs(b, max_states=150000) for b in boards]
    old_time = time.time() - t0

    t0 = time.time()
    new_results = []
    for b in boards:
        geom, state, red_idx = blocks_to_fast(b)
        new_results.append(solve_bfs_fast(geom, state, red_idx, max_states=150000))
    new_time = time.time() - t0

    mismatches = sum(1 for a, c in zip(old_results, new_results) if a != c)

    print(f"  Old (dict/object) solver: {old_time:.3f}s")
    print(f"  New (bitmask) solver:     {new_time:.3f}s")
    print(f"  Speedup: {old_time/new_time:.2f}x")
    print(f"  Result mismatches (correctness check): {mismatches} / {len(boards)}")
