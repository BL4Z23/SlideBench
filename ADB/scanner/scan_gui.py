"""
Interactive scanning GUI.

Workflow:
  - Set the tier in the text box (persists until you change it).
  - Press Enter (anywhere in the window):
      * First time ever: you're shown the current screen and asked to
        click on the "Next level" button. That location is saved to
        button_calibration.json and reused every time after.
      * Every time after: it (1) taps that button via adb, (2) waits
        briefly for the app to load the next level, (3) takes a
        screenshot, (4) runs the scanner, (5) saves a JSON file for
        that puzzle labeled with the current tier, (6) draws the
        detected board on screen so you can visually confirm it read
        correctly, (7) is ready for you to press Enter again.

Requires: board calibration.json (top-left/bottom-right of the puzzle
grid) must already exist, same as the original scanner.py expects.
"""

import json
import os
import subprocess
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

from scanner import load_templates, scan_board

CALIBRATION_PATH = "calibration.json"
BUTTON_CALIB_PATH = "button_calibration.json"
OUTPUT_DIR = "../scans"
CANVAS_SIZE = 480
TAP_TO_SCREENSHOT_DELAY = 0.6  # seconds, let the app finish loading the next level
AUTO_CONTINUE_DELAY_MS = 100  # tiny gap between append and next step, just to keep the GUI responsive

BG_OUTER = "#8a6a3f"
BG_BOARD = "#5c4530"
GRID_LINE = "#4a3624"
EXIT_MARK = "#2e2115"
RED_FILL = "#c0392b"
RED_STROKE = "#8e2820"
BLOCK_FILL = "#c8894a"
BLOCK_STROKE = "#9c6530"


def adb_capture():
    raw = subprocess.check_output(["adb", "exec-out", "screencap", "-p"])
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def adb_tap(x, y):
    subprocess.check_call(["adb", "shell", "input", "tap", str(x), str(y)])


def load_button_calibration():
    if os.path.exists(BUTTON_CALIB_PATH):
        with open(BUTTON_CALIB_PATH) as f:
            return json.load(f)
    return None


def save_button_calibration(x, y):
    with open(BUTTON_CALIB_PATH, "w") as f:
        json.dump({"x": x, "y": y}, f, indent=2)


class ButtonPicker(tk.Toplevel):
    """Shown once, on first Enter press: lets the user click on a live
    screenshot to record where the 'next level' button is."""

    def __init__(self, master, screenshot_bgr, on_picked):
        super().__init__(master)
        self.title("Click the 'next level' button")
        self.on_picked = on_picked

        h, w = screenshot_bgr.shape[:2]
        self.scale = min(900 / w, 900 / h, 1.0)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)

        rgb = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (disp_w, disp_h))

        # Encode to PPM in-memory so Tkinter's PhotoImage can load it
        # without needing PIL as a dependency.
        ok, buf = cv2.imencode(".ppm", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        self._img = tk.PhotoImage(data=buf.tobytes())

        tk.Label(self, text="Click exactly where the button is on screen",
                 font=("Segoe UI", 11, "bold")).pack(pady=6)

        self.canvas = tk.Canvas(self, width=disp_w, height=disp_h, highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self._img)
        self.canvas.bind("<Button-1>", self._clicked)

    def _clicked(self, event):
        real_x = int(event.x / self.scale)
        real_y = int(event.y / self.scale)
        save_button_calibration(real_x, real_y)
        self.destroy()
        self.on_picked(real_x, real_y)


class ScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unblock Me Scanner")
        self.configure(bg="#faf8f5")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.exists(CALIBRATION_PATH):
            messagebox.showerror(
                "Missing calibration.json",
                f"'{CALIBRATION_PATH}' not found. Run the board calibration "
                "step first (top_left/bottom_right of the puzzle grid) "
                "before using this scanner."
            )

        with open(CALIBRATION_PATH) as f:
            self.board_cal = json.load(f)

        self.templates = load_templates()
        self.button_calib = load_button_calibration()

        # --- top controls ---
        top = tk.Frame(self, bg="#faf8f5")
        top.pack(fill="x", padx=16, pady=12)

        tk.Label(top, text="Tier:", bg="#faf8f5", font=("Segoe UI", 11, "bold")).pack(side="left")
        self.tier_var = tk.StringVar(value="easy")
        self.tier_entry = tk.Entry(top, textvariable=self.tier_var, width=14, font=("Segoe UI", 11))
        self.tier_entry.pack(side="left", padx=(6, 20))

        self.status = tk.Label(top, text="Press Enter to capture", bg="#faf8f5", font=("Segoe UI", 10))
        self.status.pack(side="left")

        # --- button row ---
        btns = tk.Frame(self, bg="#faf8f5")
        btns.pack(fill="x", padx=16, pady=(0, 8))

        self.scan_again_btn = tk.Button(btns, text="Scan Again", command=self.on_scan_again)
        self.scan_again_btn.pack(side="left")

        self.amend_btn = tk.Button(btns, text="Amend Last Scan", command=self.on_toggle_amend)
        self.amend_btn.pack(side="left", padx=(8, 0))

        self.save_amend_btn = tk.Button(btns, text="Save Amendment", command=self.on_save_amend, state="disabled")
        self.save_amend_btn.pack(side="left", padx=(8, 0))

        self.auto_btn = tk.Button(btns, text="Start Auto (continuous)", command=self.on_toggle_auto)
        self.auto_btn.pack(side="left", padx=(8, 0))

        self.amend_help = tk.Label(
            self,
            text="",
            bg="#faf8f5", fg="#555", font=("Segoe UI", 9), justify="left"
        )
        self.amend_help.pack(fill="x", padx=16)

        # --- board canvas (visual) ---
        wrap = tk.Frame(self, bg=BG_OUTER, padx=16, pady=16)
        wrap.pack(padx=16, pady=(0, 16))
        self.canvas = tk.Canvas(wrap, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_BOARD, highlightthickness=0)
        self.canvas.pack()

        # Enter fires the capture loop from anywhere except while typing
        # in the tier box (so you can type "medium" and hit Enter to
        # confirm the tier without accidentally triggering a capture --
        # instead, Enter in the tier box just moves focus away).
        self.tier_entry.bind("<Return>", lambda e: self.focus())
        self.bind("<Return>", self.on_enter)

        self.last_saved_path = None
        self.last_saved_tier = None
        self.last_saved_index = None
        self.current_puzzle = None       # dict currently shown/held in memory
        self.first_capture_done = False  # True once we've scanned the level that was on screen at launch
        self.auto_running = False
        self.auto_job = None
        self.last_screenshot = None      # raw BGR image behind current_puzzle
        self.amend_mode = False
        self.selected_cell = None        # (row, col) awaiting a placement key, while amending
        self.pending_orientation = None  # "H" or "V", set by pressing h/v before a digit

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.bind("<Key>", self._on_key)

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            CANVAS_SIZE / 2, CANVAS_SIZE / 2,
            text="No scan yet", fill="#d8c9b0", font=("Segoe UI", 14)
        )

    def on_enter(self, event=None):
        # Don't trigger while the user is actively typing in the tier box
        if self.focus_get() is self.tier_entry:
            return

        if self.amend_mode:
            return  # Enter is not used for capture while amending; use the buttons

        if self.auto_running:
            return  # already cycling on its own timer

        self._do_capture_step()

    def _do_capture_step(self):
        """The actual capture logic (calibration-check, first-capture-
        without-tap, else normal tap+scan). Shared by manual Enter presses
        and the auto-cycle timer so behavior is identical either way."""
        if self.button_calib is None:
            self.status.config(text="Capturing screen for button calibration...")
            self.update_idletasks()
            try:
                shot = adb_capture()
            except Exception as e:
                messagebox.showerror("adb error", f"Could not capture screen: {e}")
                self._stop_auto()  # don't keep retrying blind if adb is broken
                return
            ButtonPicker(self, shot, lambda x, y: self._on_button_picked(x, y, shot))
            return

        if not self.first_capture_done:
            self._run_capture_cycle(skip_tap=True)
            self.first_capture_done = True
            return

        self._run_capture_cycle()

    def _on_button_picked(self, x, y, shot):
        self.button_calib = {"x": x, "y": y}
        # The screenshot taken for calibration is the level that's
        # currently on screen, pre-tap -- scan it directly instead of
        # discarding it and tapping into the next level.
        self.status.config(text=f"Button set at ({x},{y}). Scanning current level...")
        self.update_idletasks()
        self._scan_and_save(shot)
        self.first_capture_done = True

    def on_toggle_auto(self):
        if self.auto_running:
            self._stop_auto()
        else:
            if self.amend_mode:
                messagebox.showinfo("Finish amending first", "Save or cancel the current amendment before starting auto-cycle.")
                return
            self.auto_running = True
            self.auto_btn.config(text="Stop", bg="#e74c3c", fg="white")
            self.scan_again_btn.config(state="disabled")
            self.amend_btn.config(state="disabled")
            self.status.config(text="Auto-cycle running -- click Stop to end.")
            self._auto_tick()

    def _stop_auto(self):
        self.auto_running = False
        if self.auto_job is not None:
            self.after_cancel(self.auto_job)
            self.auto_job = None
        self.auto_btn.config(text="Start Auto (continuous)", bg=self.cget("bg"), fg="black")
        self.scan_again_btn.config(state="normal")
        self.amend_btn.config(state="normal")
        self.status.config(text="Auto-cycle stopped.")

    def _auto_tick(self):
        if not self.auto_running:
            return
        self._do_capture_step()
        # Continue immediately once this capture is done and appended --
        # not after a fixed wait -- so the loop runs as fast as the tap/
        # screenshot/scan cycle itself allows. AUTO_CONTINUE_DELAY_MS is
        # just a tiny yield back to the GUI event loop, not a real pause.
        if self.auto_running:  # a failure inside _do_capture_step may have called _stop_auto
            self.auto_job = self.after(AUTO_CONTINUE_DELAY_MS, self._auto_tick)

    def on_scan_again(self):
        """Re-captures and re-scans the CURRENT screen (no button tap) --
        for when a scan came out wrong and you just want to retry reading
        the same puzzle, not advance to a new one. Overwrites the same
        entry in the tier's list that was last saved (or appends fresh
        if nothing's been saved yet this session)."""
        if self.amend_mode:
            messagebox.showinfo("Finish amending first", "Save or cancel the current amendment before scanning again.")
            return
        try:
            self.status.config(text="Re-capturing current screen...")
            self.update_idletasks()
            img = adb_capture()

            self.status.config(text="Re-scanning board...")
            self.update_idletasks()
            puzzle = scan_board(img, self.board_cal, self.templates)

            if self.last_saved_tier is not None:
                tier = self.last_saved_tier
                lst = self._load_tier_list(tier)
                idx = self.last_saved_index
                puzzle["id"] = f"{tier}_{idx:03d}"
                puzzle["tier"] = tier
                lst[idx] = puzzle
                out_path = self._save_tier_list(tier, lst)
            else:
                tier = self.tier_var.get().strip() or "unknown"
                lst = self._load_tier_list(tier)
                idx = len(lst)
                puzzle["id"] = f"{tier}_{idx:03d}"
                puzzle["tier"] = tier
                lst.append(puzzle)
                out_path = self._save_tier_list(tier, lst)
                self.last_saved_tier = tier
                self.last_saved_index = idx

            self.last_saved_path = out_path
            self.current_puzzle = puzzle
            self.last_screenshot = img
            self._render_board(puzzle)

            n_pieces = len(puzzle["pieces"])
            has_red = any(p["is_red"] for p in puzzle["pieces"])
            warn = "  ⚠ NO RED PIECE DETECTED" if not has_red else ""
            self.status.config(text=f"Re-scanned, overwrote #{idx} in {out_path} ({n_pieces} pieces){warn}")

        except Exception as e:
            messagebox.showerror("Scan again failed", str(e))
            self.status.config(text="Error -- see popup.")

    def on_toggle_amend(self):
        if self.current_puzzle is None:
            messagebox.showinfo("Nothing to amend", "Scan a board first before amending it.")
            return

        self.amend_mode = not self.amend_mode
        self.selected_cell = None
        self.pending_orientation = None

        if self.amend_mode:
            self.amend_btn.config(text="Cancel Amend", relief="sunken")
            self.save_amend_btn.config(state="normal")
            self.amend_help.config(
                text="Amend mode: click a piece to DELETE it. Click an empty cell, "
                     "then press h/v (orientation) then 2/3 (length) to ADD a piece there. "
                     "Press r to add the red car (always horizontal, length 2) anchored at "
                     "the clicked cell. Esc clears the current selection."
            )
            self.status.config(text="Amending -- edit the board, then click 'Save Amendment'.")
        else:
            self.amend_btn.config(text="Amend Last Scan", relief="raised")
            self.save_amend_btn.config(state="disabled")
            self.amend_help.config(text="")
            self._render_board(self.current_puzzle)  # discard unsaved edits, redraw as-saved
            self.status.config(text="Amend cancelled.")

    def on_save_amend(self):
        if self.last_saved_tier is None or self.last_saved_index is None or self.current_puzzle is None:
            return
        lst = self._load_tier_list(self.last_saved_tier)
        lst[self.last_saved_index] = self.current_puzzle
        out_path = self._save_tier_list(self.last_saved_tier, lst)
        self.on_toggle_amend()  # exits amend mode, since amend_mode is True this flips it off
        self.status.config(text=f"Saved amendment to #{self.last_saved_index} in {out_path}")

    def _on_canvas_click(self, event):
        if not self.amend_mode or self.current_puzzle is None:
            return

        grid = self.current_puzzle["grid_size"]
        cell = CANVAS_SIZE / grid
        c = int(event.x // cell)
        r = int(event.y // cell)
        if not (0 <= r < grid and 0 <= c < grid):
            return

        # Clicking a cell that's part of an existing piece deletes that piece
        hit_piece = self._piece_at(r, c)
        if hit_piece is not None:
            self.current_puzzle["pieces"].remove(hit_piece)
            self.selected_cell = None
            self.pending_orientation = None
            self._render_board(self.current_puzzle, selection=None)
            self.status.config(text=f"Deleted piece '{hit_piece['id']}'. Amend mode still on -- Save when done.")
            return

        # Otherwise select this empty cell, awaiting orientation+length keys
        self.selected_cell = (r, c)
        self.pending_orientation = None
        self._render_board(self.current_puzzle, selection=self.selected_cell)

    def _piece_at(self, r, c):
        for p in self.current_puzzle["pieces"]:
            cells = []
            if p["orientation"] == "H":
                cells = [(p["row"], p["col"] + i) for i in range(p["length"])]
            else:
                cells = [(p["row"] + i, p["col"]) for i in range(p["length"])]
            if (r, c) in cells:
                return p
        return None

    def _occupied_cells(self):
        occ = set()
        for p in self.current_puzzle["pieces"]:
            if p["orientation"] == "H":
                occ.update((p["row"], p["col"] + i) for i in range(p["length"]))
            else:
                occ.update((p["row"] + i, p["col"]) for i in range(p["length"]))
        return occ

    def _on_key(self, event):
        if not self.amend_mode or self.selected_cell is None:
            return

        key = event.keysym.lower()
        grid = self.current_puzzle["grid_size"]
        r, c = self.selected_cell

        if key == "escape":
            self.selected_cell = None
            self.pending_orientation = None
            self._render_board(self.current_puzzle, selection=None)
            return

        if key in ("h", "v"):
            self.pending_orientation = "H" if key == "h" else "V"
            self.status.config(text=f"Orientation set to {self.pending_orientation}. Press 2 or 3 for length.")
            return

        if key == "r":
            self._try_place(r, c, length=2, orientation="H", is_red=True, piece_id="R")
            return

        if key in ("2", "3") and self.pending_orientation is not None:
            self._try_place(r, c, length=int(key), orientation=self.pending_orientation, is_red=False)
            return

    def _try_place(self, r, c, length, orientation, is_red, piece_id=None):
        grid = self.current_puzzle["grid_size"]

        if is_red and any(p["is_red"] for p in self.current_puzzle["pieces"]):
            messagebox.showwarning("Only one red piece", "A red piece already exists -- delete it first.")
            return

        if orientation == "H":
            if c + length > grid:
                messagebox.showwarning("Out of bounds", "Piece doesn't fit horizontally from this cell.")
                return
            cells = [(r, c + i) for i in range(length)]
        else:
            if r + length > grid:
                messagebox.showwarning("Out of bounds", "Piece doesn't fit vertically from this cell.")
                return
            cells = [(r + i, c) for i in range(length)]

        occ = self._occupied_cells()
        if any(cell in occ for cell in cells):
            messagebox.showwarning("Overlap", "That placement overlaps an existing piece.")
            return

        if piece_id is None:
            existing_letters = {p["id"] for p in self.current_puzzle["pieces"] if p["id"] != "R"}
            i = 0
            while chr(ord("A") + i) in existing_letters:
                i += 1
            piece_id = chr(ord("A") + i)

        self.current_puzzle["pieces"].append({
            "id": piece_id, "row": r, "col": c, "length": length,
            "orientation": orientation, "is_red": is_red,
        })
        self.selected_cell = None
        self.pending_orientation = None
        self._render_board(self.current_puzzle, selection=None)
        self.status.config(text=f"Added piece '{piece_id}'. Amend mode still on -- Save when done.")

    def _tier_file_path(self, tier):
        return os.path.join(OUTPUT_DIR, f"{tier}.json")

    def _load_tier_list(self, tier):
        """Reads the accumulating list of puzzles for this tier from disk.
        Always reads fresh from disk (not from memory) so that restarting
        the app and continuing where you left off works correctly --
        e.g. if 600 boards are already saved, the next one becomes #601
        in the same file, rather than starting a new file or overwriting."""
        path = self._tier_file_path(tier)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        return []

    def _save_tier_list(self, tier, lst):
        path = self._tier_file_path(tier)
        with open(path, "w") as f:
            json.dump(lst, f, indent=2)
        return path

    def _run_capture_cycle(self, skip_tap=False):
        try:
            if not skip_tap:
                self.status.config(text="Tapping next-level button...")
                self.update_idletasks()
                adb_tap(self.button_calib["x"], self.button_calib["y"])
                time.sleep(TAP_TO_SCREENSHOT_DELAY)

            self.status.config(text="Capturing screenshot...")
            self.update_idletasks()
            img = adb_capture()

            self._scan_and_save(img)

        except Exception as e:
            messagebox.showerror("Capture failed", str(e))
            self.status.config(text="Error -- see popup. Press Enter to retry.")
            if self.auto_running:
                self._stop_auto()

    def _scan_and_save(self, img):
        """Runs the scanner on an already-captured image, appends the
        result to the current tier's accumulating list on disk, and
        updates the on-screen preview. Shared by the normal Enter flow
        and the initial (pre-tap) first-level scan."""
        tier = self.tier_var.get().strip() or "unknown"

        self.status.config(text="Scanning board...")
        self.update_idletasks()
        puzzle = scan_board(img, self.board_cal, self.templates)

        # Read whatever's already on disk for this tier (could be 0,
        # could be 600) and append this puzzle as the next entry --
        # this is what makes resuming after a restart work correctly.
        lst = self._load_tier_list(tier)
        idx = len(lst)
        puzzle["id"] = f"{tier}_{idx:03d}"
        puzzle["tier"] = tier
        lst.append(puzzle)
        out_path = self._save_tier_list(tier, lst)

        self.last_saved_path = out_path
        self.last_saved_tier = tier
        self.last_saved_index = idx
        self.current_puzzle = puzzle
        self.last_screenshot = img
        self._render_board(puzzle)

        n_pieces = len(puzzle["pieces"])
        has_red = any(p["is_red"] for p in puzzle["pieces"])
        warn = "  ⚠ NO RED PIECE DETECTED" if not has_red else ""
        self.status.config(text=f"Saved #{idx} to {out_path} ({n_pieces} pieces){warn}")

    def _render_board(self, board, selection=None):
        self.canvas.delete("all")
        grid = board["grid_size"]
        cell = CANVAS_SIZE / grid

        for i in range(grid + 1):
            p = i * cell
            self.canvas.create_line(p, 0, p, CANVAS_SIZE, fill=GRID_LINE)
            self.canvas.create_line(0, p, CANVAS_SIZE, p, fill=GRID_LINE)

        ey = board["exit_row"] * cell
        if board["exit_side"] == "right":
            self.canvas.create_rectangle(CANVAS_SIZE - 5, ey, CANVAS_SIZE, ey + cell, fill=EXIT_MARK, outline="")
        else:
            self.canvas.create_rectangle(0, ey, 5, ey + cell, fill=EXIT_MARK, outline="")

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
                x, y, x + w, y + h,
                fill=RED_FILL if piece.get("is_red") else BLOCK_FILL,
                outline=RED_STROKE if piece.get("is_red") else BLOCK_STROKE,
                width=2,
            )

        if selection is not None:
            r, c = selection
            self.canvas.create_rectangle(
                c * cell, r * cell, (c + 1) * cell, (r + 1) * cell,
                outline="#2ecc71", width=3,
            )


if __name__ == "__main__":
    ScanApp().mainloop()
