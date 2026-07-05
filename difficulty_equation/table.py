"""
pipeline.py -- everything from raw puzzle JSON to a feature table.

Combines what used to be graph_builder.py + feature_pipeline.py +
build_graph.py into one file, since they're really one job: turn puzzle
JSON into numbers (or a picture, for debugging one puzzle at a time).

USAGE

  Build a feature CSV from a folder of puzzle JSON files (or one file
  containing a list, or your tier-accumulator files):
      python3 pipeline.py features path/to/puzzles_dir_or_file.json features.csv

  Visualize one puzzle's blocking graph as a PNG (for debugging/checking
  a single scan or hand-built puzzle):
      python3 pipeline.py graph path/to/single_puzzle.json [--output out.png]

DESIGN CONSTRAINT (see graph-building section): the blocking graph is built
using ONLY one-step reachability checks -- "can this piece slide at all,
right now, given the current board?" -- never multi-step lookahead or
move simulation. This keeps the resulting features independent of any
solver/heuristic/RL method being evaluated against them later.
"""

import os
import sys
import re
import math
import glob
import json
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

import networkx as nx
import pandas as pd


def simplify_id(puzzle_id: str):
    """'starter_003' -> 3, 'Starter_003' -> 3, 'hard_027' -> 27.
    Strips the tier-prefix and leading zeros, keeping just the number.
    If the id doesn't end in digits (unexpected format), it's left as-is
    rather than silently mangled."""
    match = re.search(r'(\d+)$', str(puzzle_id))
    if match:
        return int(match.group(1))
    return puzzle_id


# ============================================================
# Board / Piece representation + loading
# ============================================================

@dataclass
class Piece:
    id: str
    row: int
    col: int
    length: int
    orientation: str  # "H" or "V"
    is_red: bool = False

    def cells(self) -> List[Tuple[int, int]]:
        if self.orientation == "H":
            return [(self.row, self.col + i) for i in range(self.length)]
        return [(self.row + i, self.col) for i in range(self.length)]


@dataclass
class Board:
    grid_size: int
    exit_row: int
    exit_side: str
    pieces: List[Piece]

    @property
    def red(self) -> Piece:
        for p in self.pieces:
            if p.is_red:
                return p
        raise ValueError("No red piece found in puzzle")

    def occupied_map(self) -> Dict[Tuple[int, int], str]:
        occ = {}
        for p in self.pieces:
            for c in p.cells():
                occ[c] = p.id
        return occ

    @staticmethod
    def from_json(obj: dict) -> "Board":
        pieces = [
            Piece(id=p["id"], row=p["row"], col=p["col"], length=p["length"],
                  orientation=p["orientation"], is_red=p.get("is_red", False))
            for p in obj["pieces"]
        ]
        red_pieces = [p for p in pieces if p.is_red]
        default_exit_row = red_pieces[0].row if red_pieces else 0
        return Board(
            grid_size=obj["grid_size"],
            exit_row=obj.get("exit_row", default_exit_row),
            exit_side=obj.get("exit_side", "right"),
            pieces=pieces,
        )


def load_puzzle_set(path: str) -> List[Tuple[str, str, Board]]:
    """Accepts a single puzzle object OR a JSON array of puzzles."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [(o["id"], o["tier"], Board.from_json(o)) for o in obj]
    return [(obj["id"], obj["tier"], Board.from_json(obj))]


def load_single_puzzle(path: str) -> dict:
    """For the `graph` subcommand -- expects exactly one puzzle."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        if len(obj) != 1:
            raise ValueError(
                f"'{path}' contains {len(obj)} puzzles -- `graph` expects exactly "
                "one. Pick a single puzzle's JSON, or extract one entry first."
            )
        obj = obj[0]
    return obj


# ============================================================
# Blocking graph construction (one-step reachability only)
# ============================================================

def can_slide(board: Board, piece: Piece, occ: Dict[Tuple[int, int], str]) -> Tuple[bool, Set[str]]:
    """Can this piece move at least one cell in any direction RIGHT NOW?
    No lookahead -- only checks the immediately adjacent cell(s)."""
    n = board.grid_size
    blockers: Set[str] = set()
    can_move = False

    if piece.orientation == "H":
        r, c0 = piece.row, piece.col
        left_cell = (r, c0 - 1)
        if c0 - 1 >= 0:
            occupant = occ.get(left_cell)
            if occupant is None:
                can_move = True
            else:
                blockers.add(occupant)
        c1 = piece.col + piece.length - 1
        right_cell = (r, c1 + 1)
        if c1 + 1 <= n - 1:
            occupant = occ.get(right_cell)
            if occupant is None:
                can_move = True
            else:
                blockers.add(occupant)
    else:  # V
        c, r0 = piece.col, piece.row
        up_cell = (r0 - 1, c)
        if r0 - 1 >= 0:
            occupant = occ.get(up_cell)
            if occupant is None:
                can_move = True
            else:
                blockers.add(occupant)
        r1 = piece.row + piece.length - 1
        down_cell = (r1 + 1, c)
        if r1 + 1 <= n - 1:
            occupant = occ.get(down_cell)
            if occupant is None:
                can_move = True
            else:
                blockers.add(occupant)

    return can_move, blockers


def red_exit_blockers(board: Board, occ: Dict[Tuple[int, int], str]) -> Set[str]:
    red = board.red
    n = board.grid_size
    blockers = set()
    if board.exit_side == "right":
        rightmost_col = red.col + red.length - 1
        for c in range(rightmost_col + 1, n):
            occupant = occ.get((red.row, c))
            if occupant is not None:
                blockers.add(occupant)
    elif board.exit_side == "left":
        for c in range(0, red.col):
            occupant = occ.get((red.row, c))
            if occupant is not None:
                blockers.add(occupant)
    else:
        raise NotImplementedError(f"exit_side {board.exit_side} not supported yet")
    return blockers


def build_blocking_graph(board: Board) -> Dict[str, Set[str]]:
    """graph[X] = set of pieces blocking X's escape. Root is 'RED_EXIT'.
    Cycles are expected (mutual blocking) -- this is a graph, not a tree."""
    occ = board.occupied_map()
    graph: Dict[str, Set[str]] = {}
    piece_by_id = {p.id: p for p in board.pieces}

    root_blockers = red_exit_blockers(board, occ)
    graph["RED_EXIT"] = root_blockers

    visited: Set[str] = set()
    frontier = list(root_blockers)

    while frontier:
        pid = frontier.pop()
        if pid in visited:
            continue
        visited.add(pid)
        piece = piece_by_id[pid]
        _, blockers = can_slide(board, piece, occ)
        graph[pid] = blockers
        for b in blockers:
            if b not in visited:
                frontier.append(b)

    return graph


# ============================================================
# Feature extraction (21 features: 12 graph-structural, 9 board-level)
# ============================================================

def dependency_entropy(G) -> float:
    """
    How EVENLY the dependency load is spread across pieces (including the
    RED_EXIT root itself) -- not entropy over move sequences, just an
    evenness score (0 to 1) of how out-degree MASS is distributed.

    Two graphs with the same node count can have very different shapes:
      Red -> A -> B -> C                (a pure chain: RED_EXIT, A, and B
                                          each carry exactly ONE dependency
                                          -- the load is spread perfectly
                                          evenly across all three)
      Red -> A, B, C   with A -> D, E    (a branching tree: RED_EXIT alone
      and B -> F                         carries 3 of the 6 total
                                          dependency-edges -- the load is
                                          concentrated on the root)

    Each node's out-degree is treated as its SHARE of the total dependency
    edges in the graph (p_i = degree_i / total_degree). We compute the
    normalized Shannon entropy (Pielou evenness) of that distribution:
      1.0 = every contributing piece carries an equal share of the load
      lower = the load is concentrated on a small number of hub pieces
    This uses Shannon's entropy formula as the computational tool, but the
    quantity it reports is evenness/concentration of dependency load, not
    entropy of a random process.
    """
    degrees = [d for _, d in G.out_degree()]
    total = sum(degrees)
    if total == 0:
        return 0.0  # no dependencies at all -- nothing to distribute

    nonzero = [d for d in degrees if d > 0]
    k = len(nonzero)
    if k <= 1:
        return 1.0  # only one piece carries any load -- trivially "even" by convention

    probs = [d / total for d in nonzero]
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(k)
    return h / h_max if h_max > 0 else 1.0


def graph_features(graph: Dict[str, set]) -> Dict[str, float]:
    G = nx.DiGraph()
    for node, blockers in graph.items():
        G.add_node(node)
        for b in blockers:
            G.add_edge(node, b)

    real_nodes = [n for n in G.nodes if n != "RED_EXIT"]
    node_count = len(real_nodes)
    root_degree = len(graph.get("RED_EXIT", set()))

    if node_count == 0:
        return {
            "node_count": 0, "max_depth": 0, "mean_depth": 0.0, "depth_variance": 0.0,
            "cycle_count": 0, "num_sccs_with_size_ge_2": 0, "largest_scc_size": 0,
            "branching_factor": 0.0, "edge_density": 0.0, "max_in_degree": 0,
            "leaf_count": 0, "root_degree": root_degree, "dependency_entropy": 0.0,
        }

    depths = []
    for n in real_nodes:
        try:
            depths.append(nx.shortest_path_length(G, "RED_EXIT", n))
        except nx.NetworkXNoPath:
            continue

    max_depth = max(depths) if depths else 0
    mean_depth = sum(depths) / len(depths) if depths else 0.0
    depth_variance = sum((d - mean_depth) ** 2 for d in depths) / len(depths) if depths else 0.0

    try:
        cycle_count = len(list(nx.simple_cycles(G)))
    except Exception:
        cycle_count = 0

    sccs = list(nx.strongly_connected_components(G))
    largest_scc_size = max((len(s) for s in sccs), default=1)
    num_sccs_with_size_ge_2 = sum(1 for s in sccs if len(s) > 1)

    total_out_degree = sum(dict(G.out_degree(real_nodes)).values())
    branching_factor = total_out_degree / node_count if node_count else 0.0

    edge_count = G.number_of_edges()
    possible_edges = node_count * (node_count - 1) if node_count > 1 else 1
    edge_density = edge_count / possible_edges

    in_degrees = dict(G.in_degree(real_nodes))
    max_in_degree = max(in_degrees.values()) if in_degrees else 0

    out_degrees = dict(G.out_degree(real_nodes))
    leaf_count = sum(1 for v in out_degrees.values() if v == 0)

    return {
        "node_count": node_count, "max_depth": max_depth, "mean_depth": mean_depth,
        "depth_variance": depth_variance, "cycle_count": cycle_count,
        "num_sccs_with_size_ge_2": num_sccs_with_size_ge_2,
        "largest_scc_size": largest_scc_size, "branching_factor": branching_factor,
        "edge_density": edge_density, "max_in_degree": max_in_degree,
        "leaf_count": leaf_count, "root_degree": root_degree,
        "dependency_entropy": dependency_entropy(G),
    }




def board_symmetry(board: Board) -> float:
    """Fraction of cells matching their mirror image (best of horiz/vert)."""
    n = board.grid_size
    occ = set(board.occupied_map().keys())
    grid = [[(r, c) in occ for c in range(n)] for r in range(n)]
    h_match = sum(1 for r in range(n) for c in range(n) if grid[r][c] == grid[r][n - 1 - c])
    v_match = sum(1 for r in range(n) for c in range(n) if grid[r][c] == grid[n - 1 - r][c])
    return max(h_match, v_match) / (n * n)


def board_features(board: Board) -> Dict[str, float]:
    red = board.red
    n = board.grid_size

    if board.exit_side == "right":
        red_distance = (n - 1) - (red.col + red.length - 1)
    elif board.exit_side == "left":
        red_distance = red.col
    else:
        red_distance = 0

    total_cells = n * n
    occupied_cells = sum(p.length for p in board.pieces)
    fill_ratio = occupied_cells / total_cells

    lengths = [p.length for p in board.pieces]
    long_piece_ratio = sum(1 for l in lengths if l >= 3) / len(lengths)
    length3_count = sum(1 for l in lengths if l >= 3)
    avg_piece_length = sum(lengths) / len(lengths)

    horiz = sum(1 for p in board.pieces if p.orientation == "H")
    orientation_ratio = horiz / len(board.pieces)

    pieces_in_red_row = sum(
        1 for p in board.pieces
        if not p.is_red and any(cell[0] == red.row for cell in p.cells())
    )

    return {
        "red_distance": red_distance, "fill_ratio": fill_ratio,
        "long_piece_ratio": long_piece_ratio, "orientation_ratio": orientation_ratio,
        "num_pieces": len(board.pieces), "pieces_in_red_row": pieces_in_red_row,
        "avg_piece_length": avg_piece_length, "length3_count": length3_count,
        "symmetry_score": board_symmetry(board),
    }


def build_feature_table(puzzle_source: str) -> pd.DataFrame:
    rows = []
    if os.path.isdir(puzzle_source):
        files = sorted(glob.glob(os.path.join(puzzle_source, "*.json")))
        puzzles = []
        for f in files:
            puzzles.extend(load_puzzle_set(f))
    else:
        puzzles = load_puzzle_set(puzzle_source)

    for pid, tier, board in puzzles:
        graph = build_blocking_graph(board)
        row = {"id": simplify_id(pid), "tier": tier}
        row.update(graph_features(graph))
        row.update(board_features(board))
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Single-puzzle graph visualization (for debugging one scan)
# ============================================================

def print_graph_summary(puzzle_id, graph):
    print(f"\n=== Blocking graph for '{puzzle_id}' ===\n")
    print("Adjacency (piece -> pieces blocking it):")
    for node in sorted(graph.keys()):
        blockers = graph[node]
        label = "RED_EXIT (path to exit)" if node == "RED_EXIT" else f"piece '{node}'"
        if blockers:
            print(f"  {label:28s} blocked by: {', '.join(sorted(blockers))}")
        else:
            print(f"  {label:28s} blocked by: (nothing -- free to move)")

    G = nx.DiGraph()
    for node, blockers in graph.items():
        G.add_node(node)
        for b in blockers:
            G.add_edge(node, b)

    real_nodes = [n for n in G.nodes if n != "RED_EXIT"]
    print(f"\nNodes (excluding RED_EXIT): {len(real_nodes)}")
    print(f"Edges: {G.number_of_edges()}")

    cycles = list(nx.simple_cycles(G))
    print(f"Cycles: {len(cycles)}")
    for c in cycles:
        print(f"  cycle: {' -> '.join(c)} -> {c[0]}")

    sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    print(f"Tangles (SCCs with size > 1): {len(sccs)}")
    for s in sccs:
        print(f"  tangle: {sorted(s)}")

    return G


def draw_graph(G, puzzle_id, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    sccs = list(nx.strongly_connected_components(G))
    tangled_nodes = set()
    for s in sccs:
        if len(s) > 1:
            tangled_nodes.update(s)

    node_colors = []
    for n in G.nodes:
        if n == "RED_EXIT":
            node_colors.append("#c0392b")
        elif n in tangled_nodes:
            node_colors.append("#e67e22")
        else:
            node_colors.append("#3d7ea6")

    pos = nx.spring_layout(G, seed=42, k=1.2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", font_size=10)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.08")

    plt.title(f"Blocking graph: {puzzle_id}\n(red = exit path, orange = tangle, blue = ordinary blocker)", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved visualization to {output_path}")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("features", help="Build a feature CSV from a folder/file of puzzles")
    p_feat.add_argument("puzzle_source", help="Directory of *.json files, or a single JSON file/list")
    p_feat.add_argument("output_csv")

    p_graph = sub.add_parser("graph", help="Visualize one puzzle's blocking graph as a PNG")
    p_graph.add_argument("puzzle_json", help="Path to a single puzzle JSON file")
    p_graph.add_argument("--output", default=None, help="Output PNG path (default: <id>_graph.png)")

    args = ap.parse_args()

    if args.command == "features":
        df = build_feature_table(args.puzzle_source)
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(df)} rows to {args.output_csv}")
        print(df.groupby("tier").size())

    elif args.command == "graph":
        obj = load_single_puzzle(args.puzzle_json)
        board = Board.from_json(obj)
        graph = build_blocking_graph(board)
        G = print_graph_summary(obj["id"], graph)
        output_path = args.output or f"{obj['id']}_graph.png"
        draw_graph(G, obj["id"], output_path)


if __name__ == "__main__":
    main()
