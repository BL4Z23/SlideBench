import csv
import json
from Game_Engine import won, valid_moves, move, decode, SHAPE

GOAL_COL = 4


def heuristic(board):
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


def idastar(initial_board):
    start = tuple(
        (piece[0], tuple(piece[1]))
        for piece in sorted(initial_board, key=lambda p: p[1])
    )

    explored = 0
    generated = 1
    path = []

    def search(board, g, threshold, seen):
        nonlocal explored, generated

        f = g + heuristic(board)
        if f > threshold:
            return f

        explored += 1

        if won(board):
            return True

        minimum = float("inf")

        for coordinates, distance in valid_moves(board):
            new_board = move(board, coordinates, distance)

            if new_board is None:
                continue

            new_g = g + 1

            if seen.get(new_board, float("inf")) <= new_g:
                continue

            seen[new_board] = new_g
            generated += 1
            path.append((coordinates, distance))

            result = search(new_board, new_g, threshold, seen)

            if result is True:
                return True

            path.pop()

            if result < minimum:
                minimum = result

        return minimum

    threshold = heuristic(start)

    while True:
        seen = {start: 0}
        result = search(start, 0, threshold, seen)

        if result is True:
            return {
                "solver": "idastar",
                "solved": True,
                "moves": len(path),
                "path": list(path),
                "explored": explored,
                "discovered": generated,
            }

        if result == float("inf"):
            return {
                "solver": "idastar",
                "solved": False,
                "moves": None,
                "path": None,
                "explored": explored,
                "discovered": generated,
            }

        threshold = result


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

    with open("idastar_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grid in boards:
            result = idastar(decode(grid))
            result["board"] = json.dumps(grid)
            result["path"] = json.dumps(result["path"])
            writer.writerow(result)