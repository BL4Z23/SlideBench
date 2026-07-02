"""
Random Sliding-Block ("Unblock Me" / Rush Hour) puzzle generator.

Approach (as requested): generate boards by RANDOM placement (not by
scrambling backward from a solved state), then validate each board with
a BFS solver. Only boards that are solvable (and not too trivial / not
absurdly hard) get kept. Run this in a loop to mass-produce puzzles.

Grid: 6x6. Red block is always horizontal, sits on row `exit_row`,
and must reach the right edge (col 5) to "win".
"""

import random
import json
from collections import deque

GRID = 6
EXIT_ROW = 2  # row index (0-based) the red block must clear, matches screenshot (3rd row)

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

    This is occupancy-driven rather than block-count-driven: it makes "nice"
    boards where e.g. 13 length-2 pieces (26 cells) or a mix like nine
    length-2 + two length-3 pieces (24 cells) are equally valid outcomes --
    whatever combination of 2s and 3s lands in the target cell range. At the
    low end (min_occupancy=12) boards can be quite open/sparse; at the high
    end (26) they're dense and cluttered.

    `red_row_avoid_prob` biases placement toward a more FAVORABLE starting
    position: with this probability, a candidate placement that would occupy
    any cell in the red block's row (EXIT_ROW -- i.e. directly in its escape
    lane) is rejected and re-rolled instead. Higher values -> more likely the
    red block's row stays clear near the start, which is what makes "easy"
    boards feel easy. Use 0.0 (no bias, pure random) for hard/very-hard
    tiers and something like 0.6-0.85 for very-easy/easy tiers.

    Returns None if it can't reach at least min_occupancy within the attempt
    budget (board too sparse to be interesting) or if red placement fails.
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
        # only offer piece lengths that still fit inside the remaining budget
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
    """Return list of (new_state_key,) reachable in one slide (any distance)."""
    occ = occupied_cells(blocks)
    moves = []
    for b in blocks:
        for cell in b.cells():
            del occ[cell]  # temporarily remove this block from occ map
        if b.horizontal:
            # slide left
            c = b.col - 1
            while c >= 0 and (b.row, c) not in occ:
                moves.append((b.id, b.row, c))
                c -= 1
            # slide right
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
    result) -- kept under this name so every existing caller (build_dataset.py,
    build_full_dataset.py, etc.) gets the speedup automatically with zero
    changes on their end."""
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
                return None  # too hard / treat as unsolved-ish, discard
    return None  # genuinely unsolvable


def board_to_dict(blocks):
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


# ---------------------------------------------------------------------------
# Multiprocessing batch generation (random-placement + BFS-validate method)
# ---------------------------------------------------------------------------

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
    import multiprocessing as mp
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
# METHOD 2: solved -> scrambled via plain BFS (full reachable-state exploration)
# Returns the move PATH (solved -> ... -> scrambled) too, useful for RL
# imitation learning / reward shaping.
# ---------------------------------------------------------------------------

def make_solved_seed(num_blocks=7):
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
# METHOD 3: solved -> scrambled via A* (targeted priority search)
# Now also returns the move path.
# ---------------------------------------------------------------------------

import heapq

def astar_solved_to_scrambled(target_depth=8, num_blocks=7, max_expansions=50000):
    seed_blocks = make_solved_seed(num_blocks)
    start_key = state_key(seed_blocks)

    def heuristic(key):
        blocks = apply_state(seed_blocks, key)
        red = next(b for b in blocks if b.is_red)
        dist_from_exit = (GRID - red.length) - red.col
        # BOUNDED heuristic: normalized to [-1, 0]. Previously this was the
        # raw -dist_from_exit, which is UNBOUNDED and could dominate the
        # abs(g - target_depth) term entirely -- causing the search to chase
        # maximally-scrambled states regardless of target depth (root cause
        # of the wide depth spread we patched earlier with a retry wrapper).
        # Now it can only act as a tie-breaker between states that are
        # otherwise equally close to the target depth, not override it.
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


# ---------------------------------------------------------------------------
# METHOD 4: solved -> scrambled via IDA* (iterative deepening A*)
# ---------------------------------------------------------------------------
# IDA* repeatedly does depth-limited DFS, raising the f-bound each round
# (rather than keeping a priority queue like A*, or expanding whole layers
# like BFS). This gives a THIRD distinct generation bias: IDA*'s search
# order is depth-first within each bound, so it tends to commit early to
# one block's moves before backtracking -- structurally different from
# both BFS's breadth-first layering and A*'s best-first ordering. Good as
# its own "IDA*-favorable" condition for your training comparison.

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
# Depth-stratified batch generation: build N training boards per generation
# method, evenly spread across a target depth range, so the three (or four)
# training sets are difficulty-matched and only differ in HOW the board was
# reached -- isolating "generation bias" as the variable you're studying.
# ---------------------------------------------------------------------------

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


GENERATORS = {
    "bfs": bfs_solved_to_scrambled,
    "astar": astar_solved_to_scrambled_tolerant,
    "idastar": ida_star_solved_to_scrambled,
}


# ---------------------------------------------------------------------------
# SOLVER-AGNOSTIC DIFFICULTY METRIC: blocking-dependency graph
# ---------------------------------------------------------------------------
# Pure geometry/constraint analysis -- no BFS/A*/IDA* path search involved.
# Build a directed graph: block X "blocks" block Y if X currently sits on a
# cell that Y would need to pass through to clear its own nearest obstacle
# in some direction. We compute:
#   - direct blockers of the red block's path to the exit
#   - for each blocker, whether ITS escape route (up/down for vertical
#     blockers in red's row) is itself blocked by other blocks -> recursion
#   - whether the resulting dependency graph contains cycles (mutual
#     lockout: A blocks B's only way out, B blocks A's only way out)
#   - mobility (free slide room) for every block, and overall density
# These combine into a single difficulty_score with no reference to any
# search algorithm's notion of "moves" or "distance".

def _block_mobility(b, occ_no_self):
    """How many empty cells this block could slide into right now (both
    directions combined). Pure occupancy check, not a search."""
    free = 0
    if b.horizontal:
        c = b.col - 1
        while c >= 0 and (b.row, c) not in occ_no_self:
            free += 1
            c -= 1
        c = b.col + b.length
        while c < GRID and (b.row, c) not in occ_no_self:
            free += 1
            c += 1
    else:
        r = b.row - 1
        while r >= 0 and (r, b.col) not in occ_no_self:
            free += 1
            r -= 1
        r = b.row + b.length
        while r < GRID and (r, b.col) not in occ_no_self:
            free += 1
            r += 1
    return free


def build_blocking_graph(blocks):
    """Directed edges: blocker_id -> blocked_id, meaning `blocker` currently
    occupies a cell in `blocked`'s row/column that sits between `blocked`
    and ITS nearest clear escape in some direction. Geometry only."""
    occ = occupied_cells(blocks)
    by_id = {b.id: b for b in blocks}
    edges = set()

    for b in blocks:
        occ_no_self = {cell: bid for cell, bid in occ.items() if bid != b.id}
        if b.horizontal:
            # scan rightwards from b's right end to the wall; first occupied
            # cell's owner is a blocker (for rightward escape)
            c = b.col + b.length
            while c < GRID:
                if (b.row, c) in occ_no_self:
                    edges.add((occ_no_self[(b.row, c)], b.id))
                    break
                c += 1
            c = b.col - 1
            while c >= 0:
                if (b.row, c) in occ_no_self:
                    edges.add((occ_no_self[(b.row, c)], b.id))
                    break
                c -= 1
        else:
            r = b.row + b.length
            while r < GRID:
                if (r, b.col) in occ_no_self:
                    edges.add((occ_no_self[(r, b.col)], b.id))
                    break
                r += 1
            r = b.row - 1
            while r >= 0:
                if (r, b.col) in occ_no_self:
                    edges.add((occ_no_self[(r, b.col)], b.id))
                    break
                r -= 1
    return edges  # set of (blocker_id, blocked_id)


def has_cycle(edges, node_ids):
    """Detect mutual-lockout cycles in the blocking graph via DFS coloring."""
    adj = {n: [] for n in node_ids}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in node_ids}

    def visit(n):
        color[n] = GRAY
        for nb in adj.get(n, []):
            if color.get(nb, WHITE) == GRAY:
                return True
            if color.get(nb, WHITE) == WHITE and visit(nb):
                return True
        color[n] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in node_ids)


def compute_difficulty_score(blocks):
    """Solver-agnostic difficulty score from pure board geometry.
    Higher = harder. No BFS/A*/IDA* involved anywhere in this function."""
    occ = occupied_cells(blocks)
    n = len(blocks)

    # 1. density
    filled_cells = len(occ)
    density = filled_cells / (GRID * GRID)

    # 2. average mobility (inverse of difficulty -- low mobility = harder)
    mobilities = []
    for b in blocks:
        occ_no_self = {cell: bid for cell, bid in occ.items() if bid != b.id}
        mobilities.append(_block_mobility(b, occ_no_self))
    avg_mobility = sum(mobilities) / n if n else 0
    red = next(b for b in blocks if b.is_red)
    red_mobility = _block_mobility(red, {c: i for c, i in occ.items() if i != red.id})

    # 3. direct blockers of red's path to the exit (purely geometric count)
    direct_blockers = 0
    c = red.col + red.length
    while c < GRID:
        if (red.row, c) in occ:
            direct_blockers += 1
        c += 1

    # 4. blocking-dependency graph + cycle detection (mutual lockouts)
    edges = build_blocking_graph(blocks)
    node_ids = [b.id for b in blocks]
    cyclic = has_cycle(edges, node_ids)
    graph_edge_count = len(edges)

    # combine into one score (weights are tunable; all components are
    # purely structural/geometric -- nothing here calls a path search)
    score = (
        density * 10
        + direct_blockers * 4
        + (1.0 / (avg_mobility + 1)) * 8
        + (1.0 / (red_mobility + 1)) * 6
        + graph_edge_count * 1.5
        + (15 if cyclic else 0)
    )

    return {
        "difficulty_score": round(score, 2),
        "density": round(density, 3),
        "avg_mobility": round(avg_mobility, 2),
        "red_mobility": red_mobility,
        "direct_blockers": direct_blockers,
        "blocking_graph_edges": graph_edge_count,
        "has_lockout_cycle": cyclic,
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
# by the solver-agnostic `difficulty_score` instead, then oversample each
# method and subsample down to equal counts per bin. This way "difficulty"
# means the same thing across all four datasets, and none of the four
# generation processes gets to define what "hard" means for the others.

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

    print("\n=== Method 1b: same, parallelized across processes ===")
    import time
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

    with open("board.json", "w") as f:
        json.dump({
            "method1_random_bfs_validated": puzzles,
            "method1_parallel_sample": par_puzzles[:5],
            "method1_unique_solution_sample": uniq_puzzles,
            "method2_bfs_scrambled": {"depth": depth, "path": path, "board": board_to_dict(board)},
            "method3_astar_scrambled": {"depth": depth2, "path": path2, "board": board_to_dict(board2)},
            "method4_idastar_scrambled": {"depth": depth3, "path": path3, "board": board_to_dict(board3)},
            "matched_training_sets_sample": {
                m: boards[:3] for m, boards in matched.items()
            },
            "difficulty_matched_4methods_sample": {
                m: boards[:3] for m, boards in matched4.items()
            },
        }, f, indent=2)
    print("\nSaved sample output -> board.json")
