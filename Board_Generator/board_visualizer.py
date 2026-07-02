"""
Puzzle board visualizer -- Python/Tkinter version.

Loads boards from a JSONL or JSON dataset file and renders them one at a
time on a Canvas, wooden-puzzle style, matching the reference game
screenshot. Prev/Next buttons (and Left/Right arrow keys) step through
the dataset.

Usage:
    python3 board_visualizer.py [path/to/boards.jsonl]

Accepts BOTH formats directly, auto-detected -- no conversion needed:
  - .jsonl  (one board record per line) -- what multimethod_dataset.py
             and tiered_difficulty_dataset.py actually produce, e.g.
             python3 board_visualizer.py full_dataset.jsonl
  - .json   (a single JSON array of records) -- the small demo sample,
             e.g. python3 board_visualizer.py boards_sample.json

If no path is given, it looks for a few common filenames next to this
script (boards_sample.json, full_dataset.jsonl, dataset_sample.jsonl, or
datasets/full_dataset.jsonl).

Pure standard library -- tkinter ships with Python, nothing to pip install.
"""

import json
import os
import sys
import tkinter as tk

CANVAS_SIZE = 480
BG_OUTER = "#8a6a3f"
BG_BOARD = "#5c4530"
GRID_LINE = "#4a3624"
EXIT_MARK = "#2e2115"
RED_FILL = "#c0392b"
RED_STROKE = "#8e2820"
BLOCK_FILL = "#c8894a"
BLOCK_STROKE = "#9c6530"


def load_boards(path):
    """Load board records from either format:
      - .jsonl (JSON Lines, one board record per line) -- what
        multimethod_dataset.py and tiered_difficulty_dataset.py actually
        produce
      - .json (a single JSON array of records) -- what the small demo
        sample uses
    Auto-detected by file extension, with a content-based fallback in case
    the extension is misleading."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            raw = f.read()
        stripped = raw.lstrip()
        if stripped.startswith("["):
            data = json.loads(raw)
        else:
            # looks like JSONL even though the extension said .json
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty list of board records")
    return data


class BoardVisualizer(tk.Tk):
    def __init__(self, boards):
        super().__init__()
        self.title("Puzzle board visualizer")
        self.configure(bg="#faf8f5")
        self.boards = boards
        self.cur = 0

        header = tk.Frame(self, bg="#faf8f5")
        header.pack(fill="x", padx=20, pady=(20, 10))

        self.idx_label = tk.Label(header, text="", font=("Segoe UI", 13, "bold"),
                                   bg="#faf8f5", fg="#2c2c2a")
        self.idx_label.pack(side="left")

        self.tier_badge = tk.Label(header, text="", font=("Segoe UI", 10),
                                    bg="#e6f1fb", fg="#0c447c", padx=10, pady=3)
        self.tier_badge.pack(side="left", padx=(10, 0))

        btn_frame = tk.Frame(header, bg="#faf8f5")
        btn_frame.pack(side="right")

        self.prev_btn = tk.Button(btn_frame, text="< Prev", command=self.prev_board,
                                   relief="solid", bd=1, padx=12, pady=4)
        self.prev_btn.pack(side="left", padx=(0, 8))

        self.next_btn = tk.Button(btn_frame, text="Next >", command=self.next_board,
                                   relief="solid", bd=1, padx=12, pady=4)
        self.next_btn.pack(side="left")

        board_wrap = tk.Frame(self, bg=BG_OUTER, padx=16, pady=16)
        board_wrap.pack(padx=20, pady=(0, 10))

        self.canvas = tk.Canvas(board_wrap, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg=BG_BOARD, highlightthickness=0)
        self.canvas.pack()

        self.stats_frame = tk.Frame(self, bg="#faf8f5")
        self.stats_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.bind("<Left>", lambda e: self.prev_board())
        self.bind("<Right>", lambda e: self.next_board())

        self.render()

    def prev_board(self):
        self.cur = (self.cur - 1) % len(self.boards)
        self.render()

    def next_board(self):
        self.cur = (self.cur + 1) % len(self.boards)
        self.render()

    def render(self):
        rec = self.boards[self.cur]
        board = rec["board"]
        grid = board["grid"]
        cell = CANVAS_SIZE / grid

        self.idx_label.config(text=f"Board {self.cur + 1} of {len(self.boards)}")
        moves = rec.get("difficulty_moves")
        self.tier_badge.config(text=f"{moves} moves to solve" if moves is not None else "")

        self.canvas.delete("all")

        for i in range(grid + 1):
            p = i * cell
            self.canvas.create_line(p, 0, p, CANVAS_SIZE, fill=GRID_LINE)
            self.canvas.create_line(0, p, CANVAS_SIZE, p, fill=GRID_LINE)

        exit_row = board["exit_row"]
        exit_y = exit_row * cell
        self.canvas.create_rectangle(
            CANVAS_SIZE - 5, exit_y, CANVAS_SIZE, exit_y + cell,
            fill=EXIT_MARK, outline=""
        )

        pad = 5
        for blk in board["blocks"]:
            x = blk["col"] * cell + pad
            y = blk["row"] * cell + pad
            if blk["horizontal"]:
                w = blk["length"] * cell - pad * 2
                h = cell - pad * 2
            else:
                w = cell - pad * 2
                h = blk["length"] * cell - pad * 2
            fill = RED_FILL if blk["is_red"] else BLOCK_FILL
            outline = RED_STROKE if blk["is_red"] else BLOCK_STROKE
            self.canvas.create_rectangle(
                x, y, x + w, y + h, fill=fill, outline=outline, width=2
            )

        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        stats = [
            ("Difficulty score", board.get("difficulty_score")),
            ("Density", board.get("density")),
            ("Direct blockers", board.get("direct_blockers")),
            ("Avg mobility", board.get("avg_mobility")),
            ("Lockout cycle", "Yes" if board.get("has_lockout_cycle") else "No"),
            ("Blocks", len(board["blocks"])),
        ]
        for label, value in stats:
            card = tk.Frame(self.stats_frame, bg="#f0ede6", padx=12, pady=8)
            card.pack(side="left", expand=True, fill="both", padx=4)
            tk.Label(card, text=label, font=("Segoe UI", 9), bg="#f0ede6",
                     fg="#777").pack(anchor="w")
            tk.Label(card, text=str(value), font=("Segoe UI", 13, "bold"),
                     bg="#f0ede6", fg="#2c2c2a").pack(anchor="w")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # try a few sensible candidates instead of assuming one exact layout
        candidates = [
            os.path.join(script_dir, "boards_sample.json"),
            os.path.join(script_dir, "full_dataset.jsonl"),
            os.path.join(script_dir, "datasets", "full_dataset.jsonl"),
            os.path.join(script_dir, "dataset_sample.jsonl"),
        ]
        path = next((c for c in candidates if os.path.exists(c)), candidates[0])

    if not os.path.exists(path):
        print(f"Could not find board data at: {path}")
        print("Pass a path to a board dataset file, e.g.:")
        print("  python3 board_visualizer.py full_dataset.jsonl")
        print("  python3 board_visualizer.py boards_sample.json")
        sys.exit(1)

    boards = load_boards(path)
    app = BoardVisualizer(boards)
    app.mainloop()


if __name__ == "__main__":
    main()
