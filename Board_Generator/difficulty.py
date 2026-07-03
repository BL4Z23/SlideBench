"""
SOLVER-AGNOSTIC DIFFICULTY METRIC: blocking-dependency graph.

Split out of puzzle_core.py into its own module so difficulty scoring has
no coupling to any particular generation method (random / bfs / astar /
idastar) and can be imported on its own, tested on its own, or swapped out
independently of how boards are actually produced.

Pure geometry/constraint analysis -- no BFS/A*/IDA* path search involved
anywhere in this file. Given a board, we build a directed graph: block X
"blocks" block Y if X currently sits on a cell that Y would need to pass
through to clear its own nearest obstacle in some direction. From that we
compute:
  - direct blockers of the red block's path to the exit
  - for each blocker, whether ITS escape route is itself blocked by other
    blocks -> recursive dependency, captured as graph edges
  - whether the resulting dependency graph contains cycles (mutual lockout:
    A blocks B's only way out, B blocks A's only way out)
  - mobility (free slide room) for every block, and overall board density
These combine into a single difficulty_score with no reference to any
search algorithm's notion of "moves" or "distance" -- so the exact same
function is used to score boards regardless of which method (random / bfs /
astar / idastar) generated them, keeping the label independent of the
generator.
"""

from puzzle_manager import GRID, occupied_cells


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
    Higher = harder. No BFS/A*/IDA* involved anywhere in this function --
    this is what keeps difficulty comparable across boards regardless of
    which method generated them."""
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
