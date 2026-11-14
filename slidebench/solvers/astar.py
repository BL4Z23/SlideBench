import csv
import heapq
import json
from itertools import count
from slidebench.engine import won, valid_moves, move, decode,encode, SHAPE
GOAL_COL = 4


def heuristic(board):
    board = decode(board)
    red = next(p for p in board if p[0] == "R")
    r, c = red[1]
    if c == GOAL_COL:
        return 0
    blockers = set()
    for i, (ptype, (pr, pc)) in enumerate(board):
        if ptype == "R":
            continue
        length, orient = SHAPE[ptype]
        cells = [(pr, pc + k) if orient == "H" else (pr + k, pc)
                 for k in range(length)]
        if any(cr == r and cc > c + 1 for cr, cc in cells):
            blockers.add(i)
    return len(blockers) + 1


def astar(initial_board):
    start = tuple(
        (piece[0], tuple(piece[1]))
        for piece in sorted(initial_board, key=lambda p: p[1])
    )

    tie = count()
    heap = [(heuristic(encode(start)), 0, next(tie), start)]
    g_best = {start: 0}
    parent = {start: None}
    explored = 0

    while heap:
        f, g, _, current_board = heapq.heappop(heap)

        if g > g_best[current_board]:
            continue

        explored += 1

        if won(current_board):
            path = []
            current = current_board

            while parent[current] is not None:
                previous, used_move = parent[current]
                path.append(used_move)
                current = previous

            path.reverse()
            return {
                "solver": "astar",
                "solved": True,
                "moves": len(path),
                "path": path,
                "explored": explored,
                "discovered": len(g_best),
            }

        for coordinates, distance in valid_moves(current_board):
            new_board = move(current_board, coordinates, distance)

            if new_board is None:
                continue

            new_g = g + 1

            if new_g < g_best.get(new_board, float("inf")):
                g_best[new_board] = new_g
                parent[new_board] = (current_board, (coordinates, distance))
                heapq.heappush(
                    heap, (new_g + heuristic(encode(new_board)), new_g, next(tie), new_board)
                )

    return {
        "solver": "astar",
        "solved": False,
        "moves": None,
        "path": None,
        "explored": explored,
        "discovered": len(g_best),
    }


if __name__ == '__main__':
    boards = [
        [
            [3, 3, 3, 0, 0, 5],
            [0, 0, 5, 0, 0, 5],
            [1, 1, 5, 0, 0, 5],
            [4, 0, 5, 0, 2, 2],
            [4, 0, 0, 0, 4, 0],
            [3, 3, 3, 0, 4, 0]
        ],
    ]

    fields = ["board", "solver", "solved", "moves", "explored", "discovered", "path"]

    with open("astar_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grid in boards:
            result = astar(decode(grid))
            result["board"] = json.dumps(grid)
            result["path"] = json.dumps(result["path"])
            writer.writerow(result)