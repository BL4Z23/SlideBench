import csv
import heapq
import json
from itertools import count

from slidebench.engine import won, valid_moves, move, decode,encode, SHAPE
from MLP_Model import calculate_MLP_heuristic

Model_Path = "../../../data/Heuristic_Data/model_lr-e-3_l-25.pth"

def astar_with_lh(initial_board, learned_heuristic):
    start = tuple(
        (piece[0], tuple(piece[1]))
        for piece in sorted(initial_board, key=lambda p: p[1])
    )

    tie = count()
    heap = [(learned_heuristic(encode(start),Model_Path), 0, next(tie), start)]
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
                    heap, (new_g + learned_heuristic(encode(new_board),Model_Path), new_g, next(tie), new_board)
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
            result = astar_with_lh(decode(grid), calculate_MLP_heuristic)
            result["board"] = json.dumps(grid)
            result["path"] = json.dumps(result["path"])
            writer.writerow(result)