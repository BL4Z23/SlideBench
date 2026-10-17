
"""
SlideBench | Puzzle Explorer

A polished Tkinter viewer for sliding-puzzle datasets.

Usage:
    python board_visualizer.py [path/to/boards.jsonl]

Supported formats:
    .jsonl  One board record per line
    .json   A JSON array of board records

If no path is supplied, common dataset locations are checked
relative to this script.

Requires only the Python standard library.
"""

import json
import os
import sys
import tkinter as tk
from tkinter import messagebox


# ─────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────

APP_BG = "#F5F3EF"
PANEL_BG = "#FFFFFF"
SURFACE_BG = "#FBF9F5"

TEXT = "#252822"
TEXT_MUTED = "#85867F"
TEXT_LIGHT = "#B8B8B0"

ACCENT = "#B8753D"
ACCENT_DARK = "#88502A"
ACCENT_PALE = "#F5E8DA"

BOARD_WOOD = "#735239"
BOARD_INNER = "#493728"
BOARD_GRID = "#5E4937"
BOARD_FRAME = "#D9C5A8"

BLOCK_WOOD = "#E7B878"
BLOCK_LIGHT = "#F2D09B"
BLOCK_EDGE = "#B77D43"
BLOCK_SHADOW = "#302317"

TARGET_RED = "#D95B4D"
TARGET_LIGHT = "#F17E6E"
TARGET_EDGE = "#A83E34"

EXIT_COLOR = "#F0C66C"
EXIT_DARK = "#30251B"

FONT = "Segoe UI"

BOARD_SIZE = 480
BOARD_PADDING = 16


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

def load_boards(path):
    """Load records from JSON or JSONL."""
    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8-sig") as file:
        raw = file.read()

    if not raw.strip():
        raise ValueError("The dataset file is empty.")

    if ext == ".jsonl":
        records = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    else:
        stripped = raw.lstrip()

        if stripped.startswith("["):
            records = json.loads(raw)
        else:
            records = [
                json.loads(line)
                for line in raw.splitlines()
                if line.strip()
            ]

    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty list of board records.")

    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(
            record.get("board"), dict
        ):
            raise ValueError(
                f"Record {index + 1} does not contain a valid board."
            )

    return records


# ─────────────────────────────────────────────────────────────
# Canvas drawing helpers
# ─────────────────────────────────────────────────────────────

def rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    """Draw a rounded rectangle using standard Tkinter primitives."""
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))

    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]

    return canvas.create_polygon(
        points,
        smooth=True,
        splinesteps=12,
        **kwargs
    )


def format_value(value, decimals=2):
    """Format numbers cleanly without unnecessary trailing zeros."""
    if value is None:
        return "—"

    if isinstance(value, float):
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    if isinstance(value, bool):
        return "Yes" if value else "No"

    return str(value)


# ─────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────

class BoardVisualizer(tk.Tk):
    def __init__(self, boards, dataset_path=""):
        super().__init__()

        self.boards = boards
        self.dataset_path = dataset_path
        self.cur = 0
        self.grid_size = 0

        self.title("SlideBench | Puzzle Explorer")
        self.configure(bg=APP_BG)
        self.geometry("1080x790")
        self.minsize(850, 680)

        self._build_interface()
        self.bind("<Left>", lambda event: self.prev_board())
        self.bind("<Right>", lambda event: self.next_board())
        self.bind("<Home>", lambda event: self.go_to_board(0))
        self.bind("<End>", lambda event: self.go_to_board(len(self.boards) - 1))
        self.bind("<Escape>", lambda event: self.destroy())

        self.render()

    # ── Interface ─────────────────────────────────────────────

    def _build_interface(self):
        header = tk.Frame(self, bg=APP_BG)
        header.pack(fill="x", padx=30, pady=(25, 18))

        brand = tk.Frame(header, bg=APP_BG)
        brand.pack(side="left")

        logo = tk.Canvas(
            brand, width=42, height=42,
            bg=APP_BG, highlightthickness=0
        )
        logo.pack(side="left", padx=(0, 12))

        rounded_rect(
            logo, 2, 2, 40, 40, 9,
            fill=ACCENT, outline=""
        )

        for row in range(2):
            for col in range(2):
                x = 11 + col * 11
                y = 11 + row * 11
                rounded_rect(
                    logo, x, y, x + 7, y + 7, 2,
                    fill="#FFF4E5", outline=""
                )

        title_box = tk.Frame(brand, bg=APP_BG)
        title_box.pack(side="left")

        tk.Label(
            title_box,
            text="SLIDEBENCH",
            font=(FONT, 17, "bold"),
            fg=TEXT, bg=APP_BG
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="PUZZLE EXPLORER  /  DATASET VIEWER",
            font=(FONT, 8, "bold"),
            fg=TEXT_MUTED, bg=APP_BG
        ).pack(anchor="w", pady=(2, 0))

        self.dataset_label = tk.Label(
            header,
            text="LOCAL DATASET",
            font=(FONT, 9, "bold"),
            fg=ACCENT_DARK,
            bg=ACCENT_PALE,
            padx=12, pady=7
        )
        self.dataset_label.pack(side="right", anchor="center")

        # Main workspace
        workspace = tk.Frame(self, bg=APP_BG)
        workspace.pack(
            fill="both", expand=True,
            padx=30, pady=(0, 16)
        )

        workspace.columnconfigure(0, weight=6)
        workspace.columnconfigure(1, weight=4)
        workspace.rowconfigure(0, weight=1)

        self.board_panel = tk.Frame(
            workspace,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground="#E8E4DC"
        )
        self.board_panel.grid(
            row=0, column=0, sticky="nsew",
            padx=(0, 12)
        )

        self.info_panel = tk.Frame(
            workspace,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground="#E8E4DC"
        )
        self.info_panel.grid(
            row=0, column=1, sticky="nsew",
            padx=(12, 0)
        )

        self._build_board_panel()
        self._build_info_panel()

        # Footer navigation
        footer = tk.Frame(self, bg=APP_BG)
        footer.pack(fill="x", padx=30, pady=(0, 24))

        self.position_label = tk.Label(
            footer,
            text="",
            font=(FONT, 10),
            fg=TEXT_MUTED,
            bg=APP_BG
        )
        self.position_label.pack(side="left")

        nav = tk.Frame(footer, bg=APP_BG)
        nav.pack(side="right")

        self.prev_btn = self._button(
            nav, "←   Previous", self.prev_board,
            secondary=True
        )
        self.prev_btn.pack(side="left", padx=(0, 8))

        self.next_btn = self._button(
            nav, "Next   →", self.next_board
        )
        self.next_btn.pack(side="left")

    def _build_board_panel(self):
        panel = self.board_panel

        heading = tk.Frame(panel, bg=PANEL_BG)
        heading.pack(fill="x", padx=22, pady=(20, 12))

        tk.Label(
            heading,
            text="THE PUZZLE",
            font=(FONT, 10, "bold"),
            fg=TEXT, bg=PANEL_BG
        ).pack(anchor="w")

        tk.Label(
            heading,
            text="Slide the pieces. Find the path.",
            font=(FONT, 9),
            fg=TEXT_MUTED, bg=PANEL_BG
        ).pack(anchor="w", pady=(3, 0))

        self.canvas = tk.Canvas(
            panel,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            bg=BOARD_WOOD,
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(padx=12, pady=(0, 12))

        legend = tk.Frame(panel, bg=PANEL_BG)
        legend.pack(fill="x", padx=22, pady=(0, 18))

        self._legend_item(
            legend, TARGET_RED, "Target piece"
        ).pack(side="left", padx=(0, 18))

        self._legend_item(
            legend, BLOCK_WOOD, "Other pieces"
        ).pack(side="left", padx=(0, 18))

        self._legend_item(
            legend, EXIT_COLOR, "Exit"
        ).pack(side="left")

    def _build_info_panel(self):
        panel = self.info_panel

        heading = tk.Frame(panel, bg=PANEL_BG)
        heading.pack(fill="x", padx=22, pady=(20, 15))

        tk.Label(
            heading,
            text="BOARD INSIGHTS",
            font=(FONT, 10, "bold"),
            fg=TEXT, bg=PANEL_BG
        ).pack(anchor="w")

        tk.Label(
            heading,
            text="Structure, difficulty and mobility",
            font=(FONT, 9),
            fg=TEXT_MUTED, bg=PANEL_BG
        ).pack(anchor="w", pady=(3, 0))

        # Featured difficulty card
        self.difficulty_card = tk.Frame(
            panel, bg="#F7EFE5"
        )
        self.difficulty_card.pack(
            fill="x", padx=18, pady=(0, 16)
        )

        top = tk.Frame(self.difficulty_card, bg="#F7EFE5")
        top.pack(fill="x", padx=16, pady=(14, 4))

        tk.Label(
            top,
            text="SOLVING DEPTH",
            font=(FONT, 9, "bold"),
            fg=ACCENT_DARK, bg="#F7EFE5"
        ).pack(side="left")

        tk.Label(
            top,
            text="ESTIMATED / RECORDED",
            font=(FONT, 7, "bold"),
            fg="#A58A6D", bg="#F7EFE5"
        ).pack(side="right")

        self.moves_value = tk.Label(
            self.difficulty_card,
            text="—",
            font=(FONT, 30, "bold"),
            fg=TEXT, bg="#F7EFE5"
        )
        self.moves_value.pack(anchor="w", padx=16)

        self.moves_caption = tk.Label(
            self.difficulty_card,
            text="moves to solve",
            font=(FONT, 9),
            fg=TEXT_MUTED, bg="#F7EFE5"
        )
        self.moves_caption.pack(
            anchor="w", padx=16, pady=(0, 14)
        )

        # Metric grid
        tk.Label(
            panel,
            text="STRUCTURAL METRICS",
            font=(FONT, 9, "bold"),
            fg=TEXT_MUTED, bg=PANEL_BG
        ).pack(anchor="w", padx=22, pady=(0, 10))

        self.metric_frame = tk.Frame(panel, bg=PANEL_BG)
        self.metric_frame.pack(
            fill="x", padx=18, pady=(0, 12)
        )

        self.metric_values = {}

        metrics = [
            ("difficulty_score", "Difficulty score"),
            ("density", "Board density"),
            ("direct_blockers", "Direct blockers"),
            ("avg_mobility", "Average mobility"),
            ("blocks", "Total pieces"),
            ("has_lockout_cycle", "Lockout cycle"),
        ]

        for index, (key, label) in enumerate(metrics):
            row, col = divmod(index, 2)

            card = tk.Frame(
                self.metric_frame,
                bg=SURFACE_BG,
                highlightthickness=1,
                highlightbackground="#EFECE5"
            )
            card.grid(
                row=row, column=col,
                sticky="nsew", padx=4, pady=4
            )

            tk.Label(
                card,
                text=label.upper(),
                font=(FONT, 8, "bold"),
                fg=TEXT_MUTED, bg=SURFACE_BG
            ).pack(anchor="w", padx=10, pady=(10, 3))

            value_label = tk.Label(
                card,
                text="—",
                font=(FONT, 17, "bold"),
                fg=TEXT, bg=SURFACE_BG
            )
            value_label.pack(
                anchor="w", padx=10, pady=(0, 10)
            )

            self.metric_values[key] = value_label

        self.metric_frame.columnconfigure(0, weight=1)
        self.metric_frame.columnconfigure(1, weight=1)

        # Target details
        self.details_frame = tk.Frame(
            panel, bg=PANEL_BG
        )
        self.details_frame.pack(
            fill="x", padx=22, pady=(8, 16)
        )

        tk.Frame(
            self.details_frame,
            bg="#EAE6DE", height=1
        ).pack(fill="x", pady=(0, 12))

        tk.Label(
            self.details_frame,
            text="BOARD DETAILS",
            font=(FONT, 9, "bold"),
            fg=TEXT_MUTED, bg=PANEL_BG
        ).pack(anchor="w", pady=(0, 8))

        self.board_size_label = tk.Label(
            self.details_frame,
            text="Grid size: —",
            font=(FONT, 10),
            fg=TEXT, bg=PANEL_BG
        )
        self.board_size_label.pack(anchor="w", pady=2)

        self.target_label = tk.Label(
            self.details_frame,
            text="Exit row: —",
            font=(FONT, 10),
            fg=TEXT, bg=PANEL_BG
        )
        self.target_label.pack(anchor="w", pady=2)

    # ── Small UI helpers ──────────────────────────────────────

    def _button(self, parent, label, command, secondary=False):
        bg = PANEL_BG if secondary else ACCENT
        fg = TEXT if secondary else "#FFFFFF"
        hover = "#F0EDE7" if secondary else ACCENT_DARK

        button = tk.Button(
            parent,
            text=label,
            command=command,
            font=(FONT, 10, "bold"),
            bg=bg, fg=fg,
            activebackground=hover,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=17, pady=10,
            cursor="hand2",
            highlightthickness=1 if secondary else 0,
            highlightbackground="#E2DED5"
        )

        button.bind(
            "<Enter>",
            lambda event: button.config(bg=hover)
        )
        button.bind(
            "<Leave>",
            lambda event: button.config(bg=bg)
        )

        return button

    def _legend_item(self, parent, color, label):
        item = tk.Frame(parent, bg=PANEL_BG)

        swatch = tk.Canvas(
            item, width=12, height=12,
            bg=PANEL_BG, highlightthickness=0
        )
        swatch.pack(side="left", padx=(0, 6))
        rounded_rect(
            swatch, 1, 1, 11, 11, 3,
            fill=color, outline=""
        )

        tk.Label(
            item,
            text=label,
            font=(FONT, 9),
            fg=TEXT_MUTED,
            bg=PANEL_BG
        ).pack(side="left")

        return item

    # ── Navigation ────────────────────────────────────────────

    def prev_board(self):
        self.go_to_board((self.cur - 1) % len(self.boards))

    def next_board(self):
        self.go_to_board((self.cur + 1) % len(self.boards))

    def go_to_board(self, index):
        self.cur = index
        self.render()

    # ── Board rendering ───────────────────────────────────────

    def render(self):
        record = self.boards[self.cur]
        board = record["board"]
        blocks = board["blocks"]
        grid = board["grid"]

        if not isinstance(grid, int) or grid < 1:
            raise ValueError("Board grid must be a positive integer.")

        self.grid_size = grid

        self.dataset_label.config(
            text=os.path.basename(self.dataset_path)
            if self.dataset_path else "LOCAL DATASET"
        )

        self.position_label.config(
            text=f"BOARD  {self.cur + 1:03d}  /  {len(self.boards):03d}"
        )

        moves = record.get("difficulty_moves")

        self.moves_value.config(
            text=format_value(moves, 0)
        )
        self.moves_caption.config(
            text="moves to solve"
            if moves is not None
            else "solving depth unavailable"
        )

        metrics = {
            "difficulty_score": board.get("difficulty_score"),
            "density": board.get("density"),
            "direct_blockers": board.get("direct_blockers"),
            "avg_mobility": board.get("avg_mobility"),
            "blocks": len(blocks),
            "has_lockout_cycle": (
                "Yes" if board.get("has_lockout_cycle") else "No"
            ),
        }

        for key, value in metrics.items():
            decimals = 3 if key == "avg_mobility" else 2
            if key == "blocks" or key == "has_lockout_cycle":
                decimals = 0

            self.metric_values[key].config(
                text=format_value(value, decimals)
            )

        self.board_size_label.config(
            text=f"Grid size: {grid} × {grid}"
        )
        self.target_label.config(
            text=f"Exit row: {board.get('exit_row', '—')}"
        )

        self._draw_board(board)

    def _draw_board(self, board):
        canvas = self.canvas
        canvas.delete("all")

        size = BOARD_SIZE
        grid = board["grid"]
        cell = size / grid

        # Wood frame and recessed playing surface
        rounded_rect(
            canvas, 3, 3, size - 3, size - 3, 14,
            fill=BOARD_FRAME, outline="#C8B18F", width=2
        )

        inset = 8

        rounded_rect(
            canvas, inset, inset,
            size - inset, size - inset, 9,
            fill=BOARD_WOOD, outline=""
        )

        # Subtle inset around the playing surface
        canvas.create_rectangle(
            inset + 5, inset + 5,
            size - inset - 5, size - inset - 5,
            fill=BOARD_INNER, outline=""
        )

        pad = inset + 5
        board_extent = size - pad * 2
        cell = board_extent / grid

        # Grid
        for i in range(grid + 1):
            p = pad + i * cell

            canvas.create_line(
                p, pad, p, size - pad,
                fill=BOARD_GRID, width=1
            )
            canvas.create_line(
                pad, p, size - pad, p,
                fill=BOARD_GRID, width=1
            )

        # Exit lane, positioned on the right edge
        exit_row = board.get("exit_row", 0)
        exit_y1 = pad + exit_row * cell
        exit_y2 = exit_y1 + cell

        canvas.create_rectangle(
            size - pad - 1, exit_y1 + 3,
            size - 3, exit_y2 - 3,
            fill=EXIT_COLOR, outline=""
        )

        # Direction marker
        arrow_x = size - 11
        arrow_y = (exit_y1 + exit_y2) / 2

        canvas.create_text(
            arrow_x, arrow_y,
            text="›",
            font=(FONT, max(12, int(cell * 0.42)), "bold"),
            fill=EXIT_DARK
        )

        # Draw pieces
        gap = max(3, min(6, cell * 0.055))

        for block in board["blocks"]:
            x1 = pad + block["col"] * cell + gap
            y1 = pad + block["row"] * cell + gap

            if block["horizontal"]:
                x2 = (
                    pad + (block["col"] + block["length"]) * cell
                    - gap
                )
                y2 = pad + (block["row"] + 1) * cell - gap
            else:
                x2 = pad + (block["col"] + 1) * cell - gap
                y2 = (
                    pad + (block["row"] + block["length"]) * cell
                    - gap
                )

            is_target = block.get("is_red", False)

            fill = TARGET_RED if is_target else BLOCK_WOOD
            highlight = TARGET_LIGHT if is_target else BLOCK_LIGHT
            edge = TARGET_EDGE if is_target else BLOCK_EDGE

            radius = max(3, min(9, cell * 0.14))

            # Soft drop shadow
            rounded_rect(
                canvas,
                x1 + 2, y1 + 3,
                x2 + 2, y2 + 3,
                radius,
                fill=BLOCK_SHADOW,
                outline=""
            )

            # Main body
            rounded_rect(
                canvas, x1, y1, x2, y2, radius,
                fill=fill,
                outline=edge,
                width=2
            )

            # Top highlight
            inset_highlight = max(4, cell * 0.12)

            if x2 - x1 > inset_highlight * 2:
                canvas.create_line(
                    x1 + inset_highlight,
                    y1 + 3,
                    x2 - inset_highlight,
                    y1 + 3,
                    fill=highlight,
                    width=2,
                    capstyle=tk.ROUND
                )

            # Small identifying detail on the target piece
            if is_target:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if block["horizontal"]:
                    mark_x = x2 - max(9, cell * 0.25)
                    mark_y = center_y
                else:
                    mark_x = center_x
                    mark_y = y2 - max(9, cell * 0.25)

                canvas.create_oval(
                    mark_x - 3, mark_y - 3,
                    mark_x + 3, mark_y + 3,
                    fill="#FFD1BA", outline=""
                )

        # Small board coordinate caption
        canvas.create_text(
            size / 2, size - 5,
            text=f"{grid} × {grid}",
            font=(FONT, 7, "bold"),
            fill="#E8D5B8"
        )


# ─────────────────────────────────────────────────────────────
# Application entry point
# ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        candidates = [
            os.path.join(script_dir, "boards_sample.json"),
            os.path.join(script_dir, "full_dataset.jsonl"),
            os.path.join(script_dir, "datasets", "full_dataset.jsonl"),
            os.path.join(script_dir, "dataset_sample.jsonl"),
        ]

        path = next(
            (candidate for candidate in candidates
             if os.path.isfile(candidate)),
            candidates[0]
        )

    if not os.path.isfile(path):
        print(f"Dataset not found: {path}")
        print("Usage: python board_visualizer.py path/to/boards.jsonl")
        sys.exit(1)

    try:
        boards = load_boards(path)
        app = BoardVisualizer(boards, dataset_path=path)
        app.mainloop()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("SlideBench | Dataset Error", str(exc))
        root.destroy()
        sys.exit(1)


if __name__ == "__main__":
    main()
