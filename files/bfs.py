import csv
import json
from collections import deque
from Game_Engine import *

def bfs(initial_board):
    initial_board = tuple(
        (piece[0], tuple(piece[1]))
        for piece in sorted(initial_board, key=lambda p: p[1])
    )

    queue = deque([initial_board])
    visited = {initial_board}
    parent = {initial_board: None}
    explored = 0

    while queue:
        current_board = queue.popleft()
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
                "solver": "bfs",
                "solved": True,
                "moves": len(path),
                "path": path,
                "explored": explored,
                "discovered": len(visited),
            }

        for coordinates, distance in valid_moves(current_board):
            new_board = move(current_board, coordinates, distance)

            if new_board is None or new_board in visited:
                continue

            visited.add(new_board)
            parent[new_board] = (current_board, (coordinates, distance))
            queue.append(new_board)

    return {
        "solver": "bfs",
        "solved": False,
        "moves": None,
        "path": None,
        "explored": explored,
        "discovered": len(visited),
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

    with open("bfs_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grid in boards:
            result = bfs(decode(grid))
            result["board"] = json.dumps(grid)
            result["path"] = json.dumps(result["path"])
            writer.writerow(result)