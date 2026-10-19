"""Extract only the optimal move count for sliding-block puzzles.

Usage:
    python optimal_moves.py puzzle.json
    python optimal_moves.py puzzles.json
    python optimal_moves.py path/to/puzzle_folder/

JSON may contain one puzzle or a list of puzzles. Each puzzle should have:
    grid_size, pieces
Each piece should have:
    id, row, col, length, orientation, is_red (optional)

A move is one vehicle sliding any positive distance in one direction.
Moving the red vehicle out through the exit counts as one move.
"""

import json
import sys
from collections import deque
from pathlib import Path


def load_puzzles(path):
    path = Path(path)

    if path.is_dir():
        puzzles = []
        for file in sorted(path.glob("*.json")):
            with file.open(encoding="utf-8") as f:
                data = json.load(f)
            puzzles.extend(data if isinstance(data, list) else [data])
        return puzzles

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, list) else [data]


def optimal_moves(puzzle):
    size = puzzle["grid_size"]
    pieces = puzzle["pieces"]

    red_index = next(
        (i for i, piece in enumerate(pieces) if piece.get("is_red", False)),
        None,
    )
    if red_index is None:
        raise ValueError(f"{puzzle.get('id', 'Puzzle')}: no piece has is_red=true")

    # Store each piece's variable coordinate: column for H, row for V.
    starts = tuple(
        piece["col"] if piece["orientation"] == "H" else piece["row"]
        for piece in pieces
    )

    def occupied(state):
        cells = {}
        for i, piece in enumerate(pieces):
            pos = state[i]
            if piece["orientation"] == "H":
                occupied_cells = ((piece["row"], pos + j)
                                  for j in range(piece["length"]))
            else:
                occupied_cells = ((pos + j, piece["col"])
                                  for j in range(piece["length"]))

            for cell in occupied_cells:
                cells[cell] = i
        return cells

    def red_can_exit(state, cells):
        red = pieces[red_index]
        pos = state[red_index]

        if red.get("exit_side", puzzle.get("exit_side", "right")) == "left":
            return all((red["row"], col) not in cells
                       for col in range(pos))
        return all((red["row"], col) not in cells
                   for col in range(pos + red["length"], size))

    queue = deque([(starts, 0)])
    visited = {starts}

    while queue:
        state, moves = queue.popleft()
        cells = occupied(state)

        # The red vehicle can leave the board in one additional move.
        if red_can_exit(state, cells):
            return moves + 1

        for i, piece in enumerate(pieces):
            orientation = piece["orientation"]
            pos = state[i]
            fixed = piece["row"] if orientation == "H" else piece["col"]

            # Try sliding in both directions, one cell at a time.
            for direction in (-1, 1):
                new_pos = pos + direction

                while 0 <= new_pos and new_pos + piece["length"] <= size:
                    if orientation == "H":
                        leading_cell = (
                            fixed,
                            new_pos if direction < 0
                            else new_pos + piece["length"] - 1,
                        )
                    else:
                        leading_cell = (
                            new_pos if direction < 0
                            else new_pos + piece["length"] - 1,
                            fixed,
                        )

                    blocker = cells.get(leading_cell)
                    if blocker is not None and blocker != i:
                        break

                    next_state_list = list(state)
                    next_state_list[i] = new_pos
                    next_state = tuple(next_state_list)

                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, moves + 1))

                    new_pos += direction

    return -1  # No solution exists.


def main():
    if len(sys.argv) != 2:
        print("Usage: python optimal_moves.py <puzzle.json | puzzle_folder>")
        raise SystemExit(2)

    for puzzle in load_puzzles(sys.argv[1]):
        puzzle_id = puzzle.get("id", "unknown")
        count = optimal_moves(puzzle)
        print(f"{puzzle_id}: {count}")


if __name__ == "__main__":
    main()
