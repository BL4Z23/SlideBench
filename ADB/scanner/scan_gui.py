"""Interactive Unblock Me scanner GUI.

Requires calibration.json and scanner.py in the working directory.
Uses ADB to capture the current board and advance to the next level.
"""

from __future__ import annotations

import json
import subprocess
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Callable
from tkinter import messagebox

import cv2
import numpy as np

from scanner import load_templates, scan_board


# Configuration
CALIBRATION_PATH = Path("calibration.json")
BUTTON_CALIB_PATH = Path("button_calibration.json")
OUTPUT_DIR = Path("../scans")

CANVAS_SIZE = 480
TAP_TO_SCREENSHOT_DELAY = 0.6
AUTO_CONTINUE_DELAY_MS = 100

# Board colors
BG_APP = "#faf8f5"
BG_OUTER = "#8a6a3f"
BG_BOARD = "#5c4530"
GRID_LINE = "#4a3624"
EXIT_MARK = "#2e2115"
RED_FILL = "#c0392b"
RED_STROKE = "#8e2820"
BLOCK_FILL = "#c8894a"
BLOCK_STROKE = "#9c6530"

Board = dict[str, Any]


# ============================================================
# ADB
# ============================================================

class ADBClient:
    """Handles communication with the Android device."""

    @staticmethod
    def capture() -> np.ndarray:
        raw = subprocess.check_output(
            ["adb", "exec-out", "screencap", "-p"]
        )
        image = cv2.imdecode(
            np.frombuffer(raw, np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError("Could not decode the ADB screenshot.")

        return image

    @staticmethod
    def tap(x: int, y: int) -> None:
        subprocess.check_call(
            ["adb", "shell", "input", "tap", str(x), str(y)]
        )


# ============================================================
# Calibration storage
# ============================================================

class CalibrationStore:
    """Loads and saves board and button calibration."""

    def __init__(
        self,
        board_path: Path,
        button_path: Path,
    ):
        self.board_path = board_path
        self.button_path = button_path

    def load_board(self) -> Board:
        if not self.board_path.exists():
            raise FileNotFoundError(
                f"{self.board_path} is missing. "
                "Run board calibration first."
            )

        with self.board_path.open(encoding="utf-8") as file:
            return json.load(file)

    def load_button(self) -> dict[str, int] | None:
        if not self.button_path.exists():
            return None

        with self.button_path.open(encoding="utf-8") as file:
            data = json.load(file)

        return {
            "x": int(data["x"]),
            "y": int(data["y"]),
        }

    def save_button(self, x: int, y: int) -> None:
        with self.button_path.open("w", encoding="utf-8") as file:
            json.dump({"x": x, "y": y}, file, indent=2)


# ============================================================
# Puzzle storage
# ============================================================

class PuzzleStore:
    """Persists puzzle lists, one JSON file per tier."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, tier: str) -> Path:
        # Prevent tier names from escaping the output directory.
        safe_tier = Path(tier).name.strip()

        if not safe_tier or safe_tier in {".", ".."}:
            safe_tier = "unknown"

        return self.output_dir / f"{safe_tier}.json"

    def load_tier(self, tier: str) -> list[Board]:
        path = self._path_for(tier)

        if not path.exists():
            return []

        with path.open(encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else [data]

    def save_tier(
        self,
        tier: str,
        puzzles: list[Board],
    ) -> str:
        path = self._path_for(tier)

        with path.open("w", encoding="utf-8") as file:
            json.dump(puzzles, file, indent=2)

        return str(path)

    @staticmethod
    def _identify(
        tier: str,
        index: int,
        puzzle: Board,
    ) -> Board:
        saved = dict(puzzle)
        saved["id"] = f"{tier}_{index:03d}"
        saved["tier"] = tier
        return saved

    def append(
        self,
        tier: str,
        puzzle: Board,
    ) -> tuple[str, int, Board]:
        puzzles = self.load_tier(tier)
        index = len(puzzles)
        saved = self._identify(tier, index, puzzle)

        puzzles.append(saved)
        path = self.save_tier(tier, puzzles)

        return path, index, saved

    def replace(
        self,
        tier: str,
        index: int,
        puzzle: Board,
    ) -> str:
        puzzles = self.load_tier(tier)

        if not 0 <= index < len(puzzles):
            raise IndexError(
                f"Scan index {index} does not exist "
                f"in tier '{tier}'."
            )

        puzzles[index] = self._identify(tier, index, puzzle)

        return self.save_tier(tier, puzzles)


# ============================================================
# Button calibration window
# ============================================================

class ButtonPicker(tk.Toplevel):
    """Lets the user select the next-level button on a screenshot."""

    def __init__(
        self,
        master: tk.Misc,
        screenshot: np.ndarray,
        on_picked: Callable[[int, int], None],
        calibration_store: CalibrationStore,
    ):
        super().__init__(master)

        self.title("Select Next-Level Button")
        self.on_picked = on_picked
        self.calibration_store = calibration_store

        height, width = screenshot.shape[:2]
        self.scale = min(900 / width, 900 / height, 1.0)

        display_width = int(width * self.scale)
        display_height = int(height * self.scale)

        resized = cv2.resize(
            screenshot,
            (display_width, display_height),
        )

        success, buffer = cv2.imencode(".ppm", resized)

        if not success:
            raise RuntimeError(
                "Could not prepare the calibration screenshot."
            )

        self.image = tk.PhotoImage(data=buffer.tobytes())

        tk.Label(
            self,
            text="Click the next-level button",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=6)

        self.canvas = tk.Canvas(
            self,
            width=display_width,
            height=display_height,
            highlightthickness=0,
        )
        self.canvas.pack()

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.image,
        )

        self.canvas.bind("<Button-1>", self._on_click)

    def _on_click(self, event: tk.Event) -> None:
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        self.calibration_store.save_button(x, y)
        self.destroy()
        self.on_picked(x, y)


# ============================================================
# Main application
# ============================================================

class ScanApp(tk.Tk):
    """Coordinates the GUI, scanning workflow, and board editor."""

    def __init__(self):
        super().__init__()

        self.title("Unblock Me Scanner")
        self.configure(bg=BG_APP)

        self.calibrations = CalibrationStore(
            CALIBRATION_PATH,
            BUTTON_CALIB_PATH,
        )

        try:
            self.board_cal = self.calibrations.load_board()
            self.button_calib = self.calibrations.load_button()
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Calibration Error", str(exc))
            self.destroy()
            raise SystemExit(1) from exc

        self.store = PuzzleStore(OUTPUT_DIR)
        self.adb = ADBClient()
        self.templates = load_templates()

        # Last saved scan
        self.last_saved_path: str | None = None
        self.last_saved_tier: str | None = None
        self.last_saved_index: int | None = None

        # Current board
        self.current_puzzle: Board | None = None
        self.last_screenshot: np.ndarray | None = None

        # Capture workflow
        self.first_capture_done = False
        self.auto_running = False
        self.auto_job: str | None = None

        # Board editor
        self.amend_mode = False
        self.selected_cell: tuple[int, int] | None = None
        self.pending_orientation: str | None = None

        self._build_ui()
        self._bind_events()
        self._draw_placeholder()

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------

    def _build_ui(self) -> None:
        top = tk.Frame(self, bg=BG_APP)
        top.pack(fill="x", padx=16, pady=12)

        tk.Label(
            top,
            text="Tier:",
            bg=BG_APP,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        self.tier_var = tk.StringVar(value="easy")

        self.tier_entry = tk.Entry(
            top,
            textvariable=self.tier_var,
            width=14,
            font=("Segoe UI", 11),
        )
        self.tier_entry.pack(side="left", padx=(6, 20))

        self.status = tk.Label(
            top,
            text="Press Enter to capture",
            bg=BG_APP,
            font=("Segoe UI", 10),
        )
        self.status.pack(side="left")

        buttons = tk.Frame(self, bg=BG_APP)
        buttons.pack(fill="x", padx=16, pady=(0, 8))

        self.scan_again_btn = tk.Button(
            buttons,
            text="Scan Again",
            command=self.on_scan_again,
        )
        self.scan_again_btn.pack(side="left")

        self.amend_btn = tk.Button(
            buttons,
            text="Amend Last Scan",
            command=self.on_toggle_amend,
        )
        self.amend_btn.pack(side="left", padx=(8, 0))

        self.save_amend_btn = tk.Button(
            buttons,
            text="Save Amendment",
            command=self.on_save_amend,
            state="disabled",
        )
        self.save_amend_btn.pack(side="left", padx=(8, 0))

        self.auto_btn = tk.Button(
            buttons,
            text="Start Auto (continuous)",
            command=self.on_toggle_auto,
        )
        self.auto_btn.pack(side="left", padx=(8, 0))

        self.amend_help = tk.Label(
            self,
            text="",
            bg=BG_APP,
            fg="#555",
            font=("Segoe UI", 9),
            justify="left",
        )
        self.amend_help.pack(fill="x", padx=16)

        wrap = tk.Frame(
            self,
            bg=BG_OUTER,
            padx=16,
            pady=16,
        )
        wrap.pack(padx=16, pady=(0, 16))

        self.canvas = tk.Canvas(
            wrap,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BG_BOARD,
            highlightthickness=0,
        )
        self.canvas.pack()

    def _bind_events(self) -> None:
        self.tier_entry.bind(
            "<Return>",
            lambda _event: self.focus(),
        )
        self.bind("<Return>", self.on_enter)
        self.bind("<Key>", self._on_key)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _set_status(self, message: str) -> None:
        self.status.config(text=message)
        self.update_idletasks()

    def _current_tier(self) -> str:
        return self.tier_var.get().strip() or "unknown"

    # --------------------------------------------------------
    # Capture workflow
    # --------------------------------------------------------

    def on_enter(self, _event: tk.Event | None = None) -> None:
        if self.focus_get() is self.tier_entry:
            return

        if self.amend_mode or self.auto_running:
            return

        self._do_capture_step()

    def _do_capture_step(self) -> None:
        if self.button_calib is None:
            self._set_status("Capturing screen for calibration...")

            try:
                screenshot = self.adb.capture()

                ButtonPicker(
                    self,
                    screenshot,
                    lambda x, y, shot=screenshot:
                        self._on_button_picked(x, y, shot),
                    self.calibrations,
                )
            except Exception as exc:
                messagebox.showerror("ADB Error", str(exc))
                self._stop_auto()

            return

        self._run_capture_cycle(
            skip_tap=not self.first_capture_done
        )

    def _on_button_picked(
        self,
        x: int,
        y: int,
        screenshot: np.ndarray,
    ) -> None:
        self.button_calib = {"x": x, "y": y}

        self._set_status(
            f"Button saved at ({x}, {y}). Scanning current level..."
        )

        try:
            self._scan_and_save(screenshot)
            self.first_capture_done = True
        except Exception as exc:
            self._handle_capture_error(exc)

    def _run_capture_cycle(self, skip_tap: bool = False) -> None:
        try:
            if not skip_tap:
                self._set_status("Tapping next-level button...")

                self.adb.tap(
                    self.button_calib["x"],
                    self.button_calib["y"],
                )

                time.sleep(TAP_TO_SCREENSHOT_DELAY)

            self._set_status("Capturing screenshot...")
            screenshot = self.adb.capture()

            self._scan_and_save(screenshot)
            self.first_capture_done = True

        except Exception as exc:
            self._handle_capture_error(exc)

    def _handle_capture_error(self, exc: Exception) -> None:
        messagebox.showerror("Capture Failed", str(exc))
        self._set_status(
            "Error. See popup, then press Enter to retry."
        )

        if self.auto_running:
            self._stop_auto()

    def _scan_and_save(self, screenshot: np.ndarray) -> None:
        tier = self._current_tier()

        self._set_status("Scanning board...")

        puzzle = scan_board(
            screenshot,
            self.board_cal,
            self.templates,
        )

        path, index, puzzle = self.store.append(tier, puzzle)

        self._remember_scan(
            path,
            tier,
            index,
            puzzle,
            screenshot,
        )

        self._show_saved_status("Saved", index, path, puzzle)

    def on_scan_again(self) -> None:
        """Rescan the current screen and replace the last saved entry."""
        if self.amend_mode:
            messagebox.showinfo(
                "Finish Editing",
                "Save or cancel the current amendment first.",
            )
            return

        try:
            self._set_status("Capturing current screen...")
            screenshot = self.adb.capture()

            self._set_status("Scanning board...")
            puzzle = scan_board(
                screenshot,
                self.board_cal,
                self.templates,
            )

            if (
                self.last_saved_tier is not None
                and self.last_saved_index is not None
            ):
                tier = self.last_saved_tier
                index = self.last_saved_index

                path = self.store.replace(tier, index, puzzle)

                puzzle["id"] = f"{tier}_{index:03d}"
                puzzle["tier"] = tier
            else:
                tier = self._current_tier()
                path, index, puzzle = self.store.append(
                    tier,
                    puzzle,
                )

            self._remember_scan(
                path,
                tier,
                index,
                puzzle,
                screenshot,
            )

            self._show_saved_status(
                "Re-scanned and replaced",
                index,
                path,
                puzzle,
            )

        except Exception as exc:
            messagebox.showerror("Scan Again Failed", str(exc))
            self._set_status("Error. See popup.")

    def _remember_scan(
        self,
        path: str,
        tier: str,
        index: int,
        puzzle: Board,
        screenshot: np.ndarray,
    ) -> None:
        self.last_saved_path = path
        self.last_saved_tier = tier
        self.last_saved_index = index

        self.current_puzzle = puzzle
        self.last_screenshot = screenshot

        self._render_board(puzzle)

    def _show_saved_status(
        self,
        prefix: str,
        index: int,
        path: str,
        puzzle: Board,
    ) -> None:
        pieces = puzzle.get("pieces", [])
        count = len(pieces)

        has_red = any(
            piece.get("is_red")
            for piece in pieces
        )

        warning = (
            " | WARNING: no red piece detected"
            if not has_red
            else ""
        )

        self._set_status(
            f"{prefix} #{index} to {path} "
            f"({count} pieces){warning}"
        )

    # --------------------------------------------------------
    # Continuous scanning
    # --------------------------------------------------------

    def on_toggle_auto(self) -> None:
        if self.auto_running:
            self._stop_auto()
            return

        if self.amend_mode:
            messagebox.showinfo(
                "Finish Editing",
                "Save or cancel the amendment before auto-scanning.",
            )
            return

        self.auto_running = True

        self.auto_btn.config(
            text="Stop",
            bg="#e74c3c",
            fg="white",
        )
        self.scan_again_btn.config(state="disabled")
        self.amend_btn.config(state="disabled")

        self._set_status("Auto-scan running. Click Stop to end.")
        self._auto_tick()

    def _stop_auto(self) -> None:
        self.auto_running = False

        if self.auto_job is not None:
            try:
                self.after_cancel(self.auto_job)
            except tk.TclError:
                pass

            self.auto_job = None

        self.auto_btn.config(
            text="Start Auto (continuous)",
            bg=BG_APP,
            fg="black",
        )
        self.scan_again_btn.config(state="normal")
        self.amend_btn.config(state="normal")

        self._set_status("Auto-scan stopped.")

    def _auto_tick(self) -> None:
        if not self.auto_running:
            return

        self._do_capture_step()

        if self.auto_running:
            self.auto_job = self.after(
                AUTO_CONTINUE_DELAY_MS,
                self._auto_tick,
            )

    # --------------------------------------------------------
    # Board editing
    # --------------------------------------------------------

    def on_toggle_amend(self) -> None:
        if self.current_puzzle is None:
            messagebox.showinfo(
                "Nothing to Edit",
                "Scan a board before editing it.",
            )
            return

        self.amend_mode = not self.amend_mode
        self.selected_cell = None
        self.pending_orientation = None

        if self.amend_mode:
            self.amend_btn.config(
                text="Cancel Amend",
                relief="sunken",
            )
            self.save_amend_btn.config(state="normal")

            self.amend_help.config(
                text=(
                    "Edit mode: click a piece to delete it. Click an "
                    "empty cell, then press h/v and 2/3 to add a piece. "
                    "Press r to add the red car (horizontal, length 2). "
                    "Esc clears the selection."
                )
            )

            self._set_status(
                "Editing board. Save your changes when finished."
            )

        else:
            self.amend_btn.config(
                text="Amend Last Scan",
                relief="raised",
            )
            self.save_amend_btn.config(state="disabled")
            self.amend_help.config(text="")

            self._reload_saved_puzzle()
            self._set_status("Amendment cancelled.")

    def _reload_saved_puzzle(self) -> None:
        if (
            self.last_saved_tier is None
            or self.last_saved_index is None
        ):
            return

        puzzles = self.store.load_tier(self.last_saved_tier)
        index = self.last_saved_index

        if 0 <= index < len(puzzles):
            self.current_puzzle = puzzles[index]
            self._render_board(self.current_puzzle)

    def on_save_amend(self) -> None:
        if (
            self.last_saved_tier is None
            or self.last_saved_index is None
            or self.current_puzzle is None
        ):
            return

        try:
            path = self.store.replace(
                self.last_saved_tier,
                self.last_saved_index,
                self.current_puzzle,
            )

            self.last_saved_path = path
            saved_index = self.last_saved_index

            self.on_toggle_amend()

            self._set_status(
                f"Saved amendment to #{saved_index} in {path}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Save Amendment Failed",
                str(exc),
            )

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self.amend_mode or self.current_puzzle is None:
            return

        grid = self.current_puzzle["grid_size"]
        cell_size = CANVAS_SIZE / grid

        col = int(event.x // cell_size)
        row = int(event.y // cell_size)

        if not (0 <= row < grid and 0 <= col < grid):
            return

        piece = self._piece_at(row, col)

        if piece is not None:
            self.current_puzzle["pieces"].remove(piece)

            self.selected_cell = None
            self.pending_orientation = None

            self._render_board(self.current_puzzle)

            self._set_status(
                f"Deleted piece '{piece['id']}'. Save when done."
            )
            return

        self.selected_cell = (row, col)
        self.pending_orientation = None

        self._render_board(
            self.current_puzzle,
            selection=self.selected_cell,
        )

    def _piece_at(
        self,
        row: int,
        col: int,
    ) -> Board | None:
        if self.current_puzzle is None:
            return None

        for piece in self.current_puzzle["pieces"]:
            if (row, col) in self._piece_cells(piece):
                return piece

        return None

    @staticmethod
    def _piece_cells(
        piece: Board,
    ) -> list[tuple[int, int]]:
        if piece["orientation"] == "H":
            return [
                (piece["row"], piece["col"] + offset)
                for offset in range(piece["length"])
            ]

        return [
            (piece["row"] + offset, piece["col"])
            for offset in range(piece["length"])
        ]

    def _occupied_cells(self) -> set[tuple[int, int]]:
        if self.current_puzzle is None:
            return set()

        return {
            cell
            for piece in self.current_puzzle["pieces"]
            for cell in self._piece_cells(piece)
        }

    def _on_key(self, event: tk.Event) -> None:
        if (
            not self.amend_mode
            or self.current_puzzle is None
            or self.selected_cell is None
        ):
            return

        key = event.keysym.lower()
        row, col = self.selected_cell

        if key == "escape":
            self.selected_cell = None
            self.pending_orientation = None
            self._render_board(self.current_puzzle)

        elif key in ("h", "v"):
            self.pending_orientation = key.upper()
            self._set_status(
                f"Orientation: {self.pending_orientation}. "
                "Press 2 or 3 for length."
            )

        elif key == "r":
            self._try_place(
                row,
                col,
                length=2,
                orientation="H",
                is_red=True,
                piece_id="R",
            )

        elif key in ("2", "3") and self.pending_orientation:
            self._try_place(
                row,
                col,
                length=int(key),
                orientation=self.pending_orientation,
                is_red=False,
            )

    def _try_place(
        self,
        row: int,
        col: int,
        length: int,
        orientation: str,
        is_red: bool,
        piece_id: str | None = None,
    ) -> None:
        if self.current_puzzle is None:
            return

        grid = self.current_puzzle["grid_size"]

        if is_red and any(
            piece.get("is_red")
            for piece in self.current_puzzle["pieces"]
        ):
            messagebox.showwarning(
                "Only One Red Piece",
                "A red piece already exists. Delete it first.",
            )
            return

        if orientation == "H":
            if col + length > grid:
                messagebox.showwarning(
                    "Out of Bounds",
                    "Piece does not fit horizontally from this cell.",
                )
                return

            cells = [
                (row, col + offset)
                for offset in range(length)
            ]

        else:
            if row + length > grid:
                messagebox.showwarning(
                    "Out of Bounds",
                    "Piece does not fit vertically from this cell.",
                )
                return

            cells = [
                (row + offset, col)
                for offset in range(length)
            ]

        occupied = self._occupied_cells()

        if any(cell in occupied for cell in cells):
            messagebox.showwarning(
                "Overlap",
                "That placement overlaps an existing piece.",
            )
            return

        if piece_id is None:
            existing_ids = {
                piece["id"]
                for piece in self.current_puzzle["pieces"]
                if piece["id"] != "R"
            }

            number = 0

            while chr(ord("A") + number) in existing_ids:
                number += 1

            piece_id = chr(ord("A") + number)

        self.current_puzzle["pieces"].append({
            "id": piece_id,
            "row": row,
            "col": col,
            "length": length,
            "orientation": orientation,
            "is_red": is_red,
        })

        self.selected_cell = None
        self.pending_orientation = None

        self._render_board(self.current_puzzle)

        self._set_status(
            f"Added piece '{piece_id}'. Save when done."
        )

    # --------------------------------------------------------
    # Board rendering
    # --------------------------------------------------------

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")

        self.canvas.create_text(
            CANVAS_SIZE / 2,
            CANVAS_SIZE / 2,
            text="No scan yet",
            fill="#d8c9b0",
            font=("Segoe UI", 14),
        )

    def _render_board(
        self,
        board: Board,
        selection: tuple[int, int] | None = None,
    ) -> None:
        self.canvas.delete("all")

        grid = board["grid_size"]
        cell = CANVAS_SIZE / grid

        # Grid lines
        for index in range(grid + 1):
            position = index * cell

            self.canvas.create_line(
                position,
                0,
                position,
                CANVAS_SIZE,
                fill=GRID_LINE,
            )
            self.canvas.create_line(
                0,
                position,
                CANVAS_SIZE,
                position,
                fill=GRID_LINE,
            )

        # Exit marker
        exit_y = board["exit_row"] * cell

        if board["exit_side"] == "right":
            self.canvas.create_rectangle(
                CANVAS_SIZE - 5,
                exit_y,
                CANVAS_SIZE,
                exit_y + cell,
                fill=EXIT_MARK,
                outline="",
            )
        else:
            self.canvas.create_rectangle(
                0,
                exit_y,
                5,
                exit_y + cell,
                fill=EXIT_MARK,
                outline="",
            )

        # Puzzle pieces
        padding = 5

        for piece in board["pieces"]:
            x = piece["col"] * cell + padding
            y = piece["row"] * cell + padding

            if piece["orientation"] == "H":
                width = piece["length"] * cell - 2 * padding
                height = cell - 2 * padding
            else:
                width = cell - 2 * padding
                height = piece["length"] * cell - 2 * padding

            is_red = piece.get("is_red", False)

            self.canvas.create_rectangle(
                x,
                y,
                x + width,
                y + height,
                fill=RED_FILL if is_red else BLOCK_FILL,
                outline=RED_STROKE if is_red else BLOCK_STROKE,
                width=2,
            )

        # Selected cell
        if selection is not None:
            row, col = selection

            self.canvas.create_rectangle(
                col * cell,
                row * cell,
                (col + 1) * cell,
                (row + 1) * cell,
                outline="#2ecc71",
                width=3,
            )


if __name__ == "__main__":
    ScanApp().mainloop()