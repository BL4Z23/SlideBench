
import cv2
import json
import numpy as np
import subprocess
import os
import argparse

def adb_capture():
    raw = subprocess.check_output(["adb","exec-out","screencap","-p"])
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def load_templates(path="templates"):
    names = {
        "red": "red.png",
        "h2": "2_Horizontal.png",
        "h3": "3_Horizontal.png",
        "v2": "2_Vertical.png",
        "v3": "3_Vertical.png",
    }
    templates = {}
    missing = []
    for key, fname in names.items():
        full = os.path.join(path, fname)
        img = cv2.imread(full, cv2.IMREAD_COLOR)
        if img is None:
            missing.append(full)
        templates[key] = img
    if missing:
        raise FileNotFoundError(
            "Could not load template image(s): " + ", ".join(missing) +
            f" (looked in '{path}/')"
        )
    return templates

def match(img, templ):
    return cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED).max()

def scan_board(img, cal, templates, threshold=0.9):
    """
    Runs the full piece-detection pipeline on a captured screenshot and
    returns a puzzle dict in the standard schema (id/tier left as
    placeholders for the caller to fill in).
    """
    x1, y1 = cal["top_left"]["x"], cal["top_left"]["y"]
    x2, y2 = cal["bottom_right"]["x"], cal["bottom_right"]["y"]
    board = img[y1:y2, x1:x2]

    cell = board.shape[0] / 6
    occ = [[False] * 6 for _ in range(6)]
    pieces = []
    t = templates

    def crop(r, c, l, o):
        px = int(round(c * cell))
        py = int(round(r * cell))
        if o == "H":
            return board[py:py + int(cell), px:px + int(cell * l)]
        return board[py:py + int(cell * l), px:px + int(cell)]

    # Red car: length 2, horizontal, fixed row 2 -- but can be at ANY
    # valid column (0 through 4), not just 0/1. This was a real bug:
    # the original only checked c in (0, 1), silently missing every
    # puzzle where red starts further right.
    for c in range(0, 6 - 2 + 1):
        s = match(crop(2, c, 2, "H"), t["red"])
        if s > threshold:
            pieces.append({"id": "R", "row": 2, "col": c, "length": 2,
                            "orientation": "H", "is_red": True})
            occ[2][c] = True
            occ[2][c + 1] = True
            break

    def scan(length, o, key):
        for r in range(6):
            for c in range(6):
                if o == "H":
                    if c + length > 6:
                        continue
                    if any(occ[r][c + i] for i in range(length)):
                        continue
                else:
                    if r + length > 6:
                        continue
                    if any(occ[r + i][c] for i in range(length)):
                        continue
                s = match(crop(r, c, length, o), t[key])
                if s < threshold:
                    continue
                pid = chr(ord("A") + len([p for p in pieces if p["id"] != "R"]))
                pieces.append({"id": pid, "row": r, "col": c, "length": length,
                                "orientation": o, "is_red": False})
                if o == "H":
                    for i in range(length):
                        occ[r][c + i] = True
                else:
                    for i in range(length):
                        occ[r + i][c] = True

    scan(3, "V", "v3")
    scan(3, "H", "h3")
    scan(2, "V", "v2")
    scan(2, "H", "h2")

    return {
        "id": "scan",
        "tier": "unknown",
        "grid_size": 6,
        "exit_row": 2,
        "exit_side": "right",
        "pieces": pieces,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--threshold",type=float,default=0.9)
    ap.add_argument("--output",default="board.json")
    args=ap.parse_args()

    with open("calibration.json") as f:
        cal=json.load(f)

    if args.image:
        img=cv2.imread(args.image)
    else:
        img=adb_capture()

    templates = load_templates()
    out = scan_board(img, cal, templates, args.threshold)

    with open(args.output,"w") as f:
        json.dump(out,f,indent=2)

    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
