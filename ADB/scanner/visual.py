import json
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
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in txt.splitlines()
            if line.strip()
        ]

    if isinstance(obj, list):
        return obj

    return [obj]


class Viewer(tk.Tk):
    def __init__(self, boards):
        super().__init__()

        self.boards = boards
        self.index = 0

        self.title("Rush Hour Board Visualizer")
        self.configure(bg="#faf8f5")

        top = tk.Frame(self, bg="#faf8f5")
        top.pack(fill="x", padx=20, pady=20)

        self.info = tk.Label(
            top,
            font=("Segoe UI", 13, "bold"),
            bg="#faf8f5"
        )
        self.info.pack(side="left")

        self.tier = tk.Label(
            top,
            bg="#e6f1fb",
            padx=10,
            pady=3
        )
        self.tier.pack(side="left", padx=10)

        tk.Button(
            top,
            text="< Prev",
            command=self.prev
        ).pack(side="right")

        tk.Button(
            top,
            text="Next >",
            command=self.next
        ).pack(side="right", padx=5)

        wrap = tk.Frame(
            self,
            bg=BG_OUTER,
            padx=16,
            pady=16
        )
        wrap.pack()

        self.canvas = tk.Canvas(
            wrap,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BG_BOARD,
            highlightthickness=0
        )
        self.canvas.pack()

        self.bind("<Left>", lambda e: self.prev())
        self.bind("<Right>", lambda e: self.next())

        self.render()

    def prev(self):
        self.index = (self.index - 1) % len(self.boards)
        self.render()

    def next(self):
        self.index = (self.index + 1) % len(self.boards)
        self.render()

    def render(self):
        board = self.boards[self.index]

        grid = board["grid_size"]
        cell = CANVAS_SIZE / grid

        self.info.config(
            text=f'{board.get("id","")} ({self.index+1}/{len(self.boards)})'
        )

        self.tier.config(
            text=board.get("tier", "")
        )

        self.canvas.delete("all")

        for i in range(grid + 1):
            p = i * cell

            self.canvas.create_line(
                p,
                0,
                p,
                CANVAS_SIZE,
                fill=GRID_LINE
            )

            self.canvas.create_line(
                0,
                p,
                CANVAS_SIZE,
                p,
                fill=GRID_LINE
            )

        ey = board["exit_row"] * cell

        if board["exit_side"] == "right":
            self.canvas.create_rectangle(
                CANVAS_SIZE - 5,
                ey,
                CANVAS_SIZE,
                ey + cell,
                fill=EXIT_MARK,
                outline=""
            )
        else:
            self.canvas.create_rectangle(
                0,
                ey,
                5,
                ey + cell,
                fill=EXIT_MARK,
                outline=""
            )

        pad = 5

        for piece in board["pieces"]:

            x = piece["col"] * cell + pad
            y = piece["row"] * cell + pad

            if piece["orientation"] == "H":
                w = piece["length"] * cell - pad * 2
                h = cell - pad * 2
            else:
                w = cell - pad * 2
                h = piece["length"] * cell - pad * 2

            self.canvas.create_rectangle(
                x,
                y,
                x + w,
                y + h,
                fill=RED_FILL if piece.get("is_red") else BLOCK_FILL,
                outline=RED_STROKE if piece.get("is_red") else BLOCK_STROKE,
                width=2
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python board_visualizer.py board.json")
        sys.exit(1)

    Viewer(load_boards(sys.argv[1])).mainloop()